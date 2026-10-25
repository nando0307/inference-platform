import asyncio
import io
import json
import logging
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

import httpx

from src import api
from src.inference import ContextLimitError, GenerationResult, MODEL_ID
from test_streaming import Connection, payload


class HeldDelivery(Connection):
    def __init__(self, stream):
        self.final_send_entered = asyncio.Event()
        self.release_send = asyncio.Event()
        super().__init__(stream)

    async def send(self, message):
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            self.final_send_entered.set()
            await self.release_send.wait()
        await super().send(message)


class LifecycleLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.output = io.StringIO()
        handler = logging.StreamHandler(self.output)
        handler_patch = patch.object(api.logger, "handlers", [handler])
        handler_patch.start()
        self.addCleanup(handler_patch.stop)
        self.addCleanup(handler.close)
        loader_patch = patch(
            "src.api.load_runtime", return_value=SimpleNamespace(tokenizer=Mock())
        )
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        context = api.app.router.lifespan_context(api.app)
        await context.__aenter__()
        self.addAsyncCleanup(context.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.marker = "PRIVATE-LIFECYCLE-MARKER"
        self.result = GenerationResult(self.marker + " generated text", 43, [987654321, 2], "stop")

    def records(self, request_id=None):
        records = [json.loads(line) for line in self.output.getvalue().splitlines()]
        return [r for r in records if request_id is None or r["request_id"] == request_id]

    def one(self, event, request_id=None):
        records = [r for r in self.records(request_id) if r["event"] == event]
        self.assertEqual(len(records), 1, records)
        return records[0]

    def assert_private(self):
        self.assertNotIn(self.marker, self.output.getvalue())
        self.assertNotIn("987654321", self.output.getvalue())
        self.assertNotIn("Traceback", self.output.getvalue())
        allowed = {
            "event", "request_id", "model", "stream", "http_status", "outcome",
            "duration_ms", "producer_duration_ms", "prompt_tokens", "completion_tokens",
            "error_code", "exception_type",
            "state", "queue_wait_ms", "cancellation_delay_ms", "cancellation_reason",
        }
        for record in self.records():
            self.assertLessEqual(record.keys(), allowed)
            self.assertEqual(UUID(record["request_id"]).hex, record["request_id"])
            self.assertEqual(record["model"], MODEL_ID)

    async def test_success_correlates_ids_and_counts_without_logging_content(self):
        body = payload(False)
        body["messages"][0]["content"] = self.marker
        body["max_tokens"] = 16
        with patch("src.api.generate", return_value=self.result):
            response = await self.client.post(
                "/v1/chat/completions", json=body,
                headers={"X-Request-ID": self.marker, "Authorization": "Bearer " + self.marker},
            )
        self.assertEqual(response.status_code, 200)
        request_id = response.headers["X-Request-ID"]
        self.assertEqual(response.json()["id"], "chatcmpl-" + request_id)
        self.one("request_accepted", request_id)
        self.one("request_started", request_id)
        self.assertEqual(self.one("scheduling_finished", request_id)["state"], "completed")
        delivery = self.one("request_finished", request_id)
        producer = self.one("producer_finished", request_id)
        self.assertEqual(delivery["http_status"], 200)
        self.assertFalse(delivery["stream"])
        for record in (delivery, producer):
            self.assertEqual(record["outcome"], "completed")
            self.assertEqual(record["prompt_tokens"], 43)
            self.assertEqual(record["completion_tokens"], 2)
            self.assertNotIn("error_code", record)
        self.assertGreaterEqual(delivery["duration_ms"], 0)
        self.assertGreaterEqual(producer["producer_duration_ms"], 0)
        self.assertNotIn("duration_ms", producer)
        self.assertNotIn("producer_duration_ms", delivery)
        self.assert_private()

    async def test_timing_includes_body_validation_and_waits_for_final_send(self):
        for stream, invalid in ((False, False), (True, False), (False, True)):
            with self.subTest(stream=stream, invalid=invalid):
                self.output.seek(0)
                self.output.truncate()
                clock = [100.0]

                def generate(*args, streamer, on_prepared, **kwargs):
                    if on_prepared is not None:
                        on_prepared()
                        streamer.on_finalized_text(self.result.text, stream_end=True)
                    return self.result

                with patch("src.api.perf_counter", side_effect=lambda: clock[0]), patch("src.api.generate", side_effect=generate):
                    connection = HeldDelivery(stream)
                    body_message = connection.incoming.get_nowait()
                    if invalid:
                        body_message["body"] = b"{}"
                    try:
                        # Enter middleware and wait for the body, before validation.
                        await asyncio.sleep(0)
                        clock[0] = 102.0
                        connection.incoming.put_nowait(body_message)
                        await asyncio.wait_for(connection.final_send_entered.wait(), 2)
                        self.assertFalse(any(r["event"] == "request_finished" for r in self.records()))
                        if not invalid:
                            self.assertEqual(self.one("producer_finished")["outcome"], "completed")
                        clock[0] = 107.0
                    finally:
                        connection.release_send.set()
                        await asyncio.wait_for(connection.task, 3)
                delivery = self.one("request_finished")
                self.assertEqual(delivery["duration_ms"], 7000.0)
                self.assertEqual(delivery["outcome"], "rejected" if invalid else "completed")
                self.assertEqual(delivery["http_status"], 422 if invalid else 200)
                self.assert_private()

    async def test_validation_logs_safe_metadata_and_one_correlated_rejection(self):
        cases = [
            ({**payload(), "model": self.marker}, True, "invalid_request"),
            ({**payload(), "stream": self.marker}, None, "invalid_request"),
            ({**payload(), self.marker: self.marker}, True, "invalid_request"),
            ({**payload(False), "top_p": 0.9}, False, "invalid_parameter_combination"),
            (("{\"" + self.marker + "\":").encode(), None, "invalid_request"),
            (b"\xff", None, "invalid_request"),
        ]
        ids = set()
        for body, stream, code in cases:
            with self.subTest(body=body):
                response = await self.client.post(
                    "/v1/chat/completions",
                    content=body if isinstance(body, bytes) else json.dumps(body),
                    headers={"Content-Type": "application/json", "X-Request-ID": self.marker},
                )
                self.assertEqual(response.status_code, 422)
                request_id = response.headers["X-Request-ID"]
                self.assertNotIn(request_id, ids)
                ids.add(request_id)
                self.assertEqual(response.json()["error"]["request_id"], request_id)
                self.assertEqual(len(self.records(request_id)), 1)
                record = self.one("request_finished", request_id)
                self.assertEqual(record["outcome"], "rejected")
                self.assertEqual(record["error_code"], code)
                self.assertEqual(record["http_status"], 422)
                self.assertIs(record["stream"], stream)
                self.assertNotIn("prompt_tokens", record)
                self.assertNotIn("completion_tokens", record)
        self.assert_private()

    async def test_busy_and_unavailable_are_rejections_without_producer_events(self):
        with (
            patch.object(api.app.state, "active_job", object()),
            patch.object(api.app.state, "pending_jobs", [object(), object()]),
        ):
            busy = await self.client.post("/v1/chat/completions", json=payload(False))
        with patch.object(api.app.state, "shutting_down", True):
            unavailable = await self.client.post("/v1/chat/completions", json=payload())
        for response, status, code in ((busy, 429, "server_busy"), (unavailable, 503, "not_ready")):
            self.assertEqual(response.status_code, status)
            records = self.records(response.headers["X-Request-ID"])
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["outcome"], "rejected")
            self.assertEqual(records[0]["error_code"], code)
            self.assertEqual(records[0]["http_status"], status)
            self.assertNotIn("completion_tokens", records[0])
        self.assert_private()

    async def test_preparation_and_generation_failure_logs_recover_without_stale_metadata(self):
        for stream in (False, True):
            for error, status, outcome, code in (
                (ContextLimitError("Prompt tokens (2040) plus output (100) exceed 2048."), 400, "rejected", "context_length_exceeded"),
                (RuntimeError(self.marker), 500, "failed", "generation_failed"),
            ):
                with self.subTest(stream=stream, status=status):
                    with patch("src.api.generate", side_effect=error):
                        response = await self.client.post("/v1/chat/completions", json=payload(stream))
                    self.assertEqual(response.status_code, status)
                    request_id = response.headers["X-Request-ID"]
                    self.one("request_accepted", request_id)
                    for event in ("producer_finished", "request_finished"):
                        record = self.one(event, request_id)
                        self.assertEqual(record["outcome"], outcome)
                        self.assertEqual(record["error_code"], code)
                        self.assertNotIn("prompt_tokens", record)
                        self.assertNotIn("completion_tokens", record)
                        if status == 500:
                            self.assertEqual(record["exception_type"], "RuntimeError")
                    self.assertEqual(self.one("request_finished", request_id)["http_status"], status)
        with patch("src.api.generate", return_value=self.result):
            recovered = await self.client.post("/v1/chat/completions", json=payload(False))
        record = self.one("request_finished", recovered.headers["X-Request-ID"])
        self.assertEqual(record["outcome"], "completed")
        self.assertNotIn("error_code", record)
        self.assertNotIn("exception_type", record)
        self.assert_private()

    async def test_submission_failure_finishes_admitted_job_without_producer_event(self):
        with patch.object(api.app.state.executor, "submit", side_effect=RuntimeError(self.marker)):
            response = await self.client.post("/v1/chat/completions", json=payload())
        self.assertEqual(response.status_code, 500)
        self.one("request_accepted")
        scheduling = self.one("scheduling_finished")
        self.assertEqual(scheduling["state"], "failed")
        self.assertIsNone(scheduling["queue_wait_ms"])
        self.assertIsNone(scheduling["cancellation_delay_ms"])
        self.assertFalse(any(record["event"] == "producer_finished" for record in self.records()))
        record = self.one("request_finished")
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(record["exception_type"], "RuntimeError")
        self.assertEqual(record["error_code"], "generation_failed")
        self.assertNotIn("completion_tokens", record)
        self.assert_private()

    async def test_postheader_failure_is_failed_even_with_http_200(self):
        release = Event()

        def generate(*args, streamer, on_prepared, **kwargs):
            on_prepared()
            streamer.on_finalized_text(self.marker)
            if not release.wait(5):
                raise TimeoutError("Producer was not released")
            raise RuntimeError(self.marker)

        with patch("src.api.generate", side_effect=generate):
            connection = Connection()
            try:
                await asyncio.wait_for(connection.content_sent.wait(), 2)
                self.assertFalse(any(r["event"] == "request_finished" for r in self.records()))
            finally:
                release.set()
                await asyncio.wait_for(connection.task, 3)
        record = self.one("request_finished")
        self.assertEqual(record["http_status"], 200)
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(record["exception_type"], "RuntimeError")
        self.assertEqual(record["error_code"], "generation_failed")
        self.assertNotIn("completion_tokens", record)
        self.assertEqual(connection.events()[-1], "[DONE]")
        self.assertEqual(connection.events()[-2]["error"]["request_id"], record["request_id"])
        self.assert_private()

    async def test_disconnect_and_cancellation_end_delivery_before_producer(self):
        for stream, spec, fail, cancel in (
            (False, "2.4", False, False),
            (True, "2.3", True, False),
            (True, "2.4", False, False),
            (False, "2.4", True, True),
        ):
            with self.subTest(stream=stream, spec=spec, fail=fail, cancel=cancel):
                self.output.seek(0)
                self.output.truncate()
                started = Event()
                release = Event()
                clock = [100.0]

                def generate(*args, streamer, on_prepared, **kwargs):
                    started.set()
                    if streamer is not None:
                        on_prepared()
                        streamer.on_finalized_text(self.marker)
                    if not release.wait(5):
                        raise TimeoutError("Producer was not released")
                    if fail:
                        raise RuntimeError(self.marker)
                    if streamer is not None:
                        streamer.on_finalized_text("", stream_end=True)
                    return self.result

                with (
                    patch("src.api.perf_counter", side_effect=lambda: clock[0]),
                    patch("src.inference.perf_counter", side_effect=lambda: clock[0]),
                    patch("src.api.generate", side_effect=generate),
                ):
                    connection = Connection(stream, spec)
                    producer = None
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 2))
                        if stream:
                            await asyncio.wait_for(connection.content_sent.wait(), 2)
                        producer = next(iter(api.app.state.producer_futures))
                        clock[0] = 105.0
                        if cancel:
                            connection.task.cancel()
                            with self.assertRaises(asyncio.CancelledError):
                                await connection.task
                        else:
                            connection.disconnect()
                            await asyncio.wait_for(connection.task, 2)
                        delivery = self.one("request_finished")
                        self.assertEqual(delivery["outcome"], "disconnected")
                        self.assertEqual(delivery["duration_ms"], 5000.0)
                        self.assertEqual(delivery["http_status"], 200 if stream else None)
                        self.assertNotIn("completion_tokens", delivery)
                        self.assertFalse(any(r["event"] == "producer_finished" for r in self.records()))
                        self.assertIsNotNone(api.app.state.active_job)
                        self.assertFalse(producer.done())
                        clock[0] = 112.0
                    finally:
                        release.set()
                        await asyncio.gather(connection.task, return_exceptions=True)
                        if producer is not None:
                            await asyncio.gather(producer, return_exceptions=True)
                worker = self.one("producer_finished")
                self.assertEqual(worker["request_id"], delivery["request_id"])
                self.assertEqual(worker["producer_duration_ms"], 12000.0)
                self.assertEqual(worker["outcome"], "failed" if fail else "canceled")
                if fail:
                    self.assertEqual(worker["exception_type"], "RuntimeError")
                    self.assertNotIn("completion_tokens", worker)
                else:
                    self.assertEqual(worker["error_code"], "request_canceled")
                    self.assertNotIn("completion_tokens", worker)
                self.assertEqual(self.one("request_finished"), delivery)
                self.assertEqual(worker["queue_wait_ms"], 0)
                self.assertEqual(worker["cancellation_delay_ms"], 7000.0)
                scheduling = self.one("scheduling_finished")
                self.assertEqual(scheduling["state"], "failed" if fail else "canceled")
                self.assertEqual(scheduling["cancellation_delay_ms"], 7000.0)
                self.assertEqual(
                    scheduling["cancellation_reason"],
                    "request_canceled" if cancel else "client_disconnected",
                )
                self.assertIsNone(api.app.state.active_job)
                self.assert_private()

    async def test_send_failure_is_a_single_disconnect_not_success(self):
        marker = self.marker

        class BrokenConnection(Connection):
            async def send(self, message):
                if message["type"] == "http.response.body":
                    raise OSError(marker)
                await super().send(message)

        with patch("src.api.generate", return_value=self.result):
            connection = BrokenConnection(False)
            await asyncio.wait_for(connection.task, 2)
        record = self.one("request_finished")
        self.assertEqual(record["outcome"], "disconnected")
        self.assertEqual(record["http_status"], 200)
        self.assertEqual(self.one("producer_finished")["outcome"], "completed")
        self.assert_private()

    async def test_unexpected_response_failure_is_sanitized_before_server_logging(self):
        with patch("src.api.generate", return_value=self.result), patch("src.api._usage", side_effect=RuntimeError(self.marker)):
            response = await self.client.post("/v1/chat/completions", json=payload(False))
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(self.marker, response.text)
        record = self.one("request_finished")
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(record["exception_type"], "RuntimeError")
        self.assertEqual(record["request_id"], response.json()["error"]["request_id"])
        self.assertEqual(self.one("producer_finished")["outcome"], "completed")
        self.assert_private()

    async def test_queue_wait_excludes_body_parsing_and_canceled_waiter_has_no_worker_metrics(self):
        started, release = Event(), Event()
        clock = [100.0]

        def held(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Producer was not released")
            return self.result

        with (
            patch("src.api.perf_counter", side_effect=lambda: clock[0]),
            patch("src.inference.perf_counter", side_effect=lambda: clock[0]),
            patch("src.api.generate", side_effect=held),
        ):
            request = api.ChatCompletionRequest.model_validate(payload(False))
            active_log = api._RequestLog(90.0)
            active = api.submit_inference(api.app, request, request_log=active_log)
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                clock[0] = 102.0
                canceled_log = api._RequestLog(91.0)
                canceled = api.submit_inference(api.app, request, request_log=canceled_log)
                clock[0] = 103.0
                queued_log = api._RequestLog(92.0)
                queued = api.submit_inference(api.app, request, request_log=queued_log)
                clock[0] = 105.0
                api.cancel_inference(api.app, canceled)
                api.cancel_inference(api.app, canceled)
                canceled_record = self.one("scheduling_finished", canceled_log.request_id)
                self.assertEqual(canceled_record["state"], "canceled")
                self.assertEqual(canceled_record["cancellation_reason"], "client_disconnected")
                self.assertIsNone(canceled_record["queue_wait_ms"])
                self.assertIsNone(canceled_record["cancellation_delay_ms"])
                self.assertFalse(any(
                    record["event"] in {"producer_finished", "request_started"}
                    for record in self.records(canceled_log.request_id)
                ))
                clock[0] = 110.0
            finally:
                release.set()
                await active.result
            await queued.result
        queued_record = self.one("scheduling_finished", queued_log.request_id)
        self.assertEqual(queued_record["state"], "completed")
        self.assertEqual(queued_record["queue_wait_ms"], 7000.0)
        self.assertIsNone(queued_record["cancellation_delay_ms"])
        self.assertEqual(self.one("request_started", queued_log.request_id)["queue_wait_ms"], 7000.0)
        self.assert_private()

    async def test_ordinary_send_deadline_records_failure_not_success(self):
        with (
            patch("src.api.generate", return_value=self.result),
            patch("src.api.STREAM_SEND_TIMEOUT", 0.05),
        ):
            connection = HeldDelivery(False)
            await asyncio.wait_for(connection.final_send_entered.wait(), 2)
            await asyncio.wait_for(connection.task, 2)
        delivery = self.one("request_finished")
        self.assertEqual(delivery["http_status"], 200)
        self.assertEqual(delivery["outcome"], "failed")
        self.assertEqual(delivery["error_code"], "slow_consumer")
        self.assertEqual(self.one("producer_finished")["outcome"], "completed")
        self.assert_private()


if __name__ == "__main__":
    unittest.main()
