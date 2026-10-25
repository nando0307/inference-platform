import asyncio
import gc
import json
import unittest

from threading import Event, Timer, get_ident
from unittest.mock import patch
from uuid import UUID

import httpx
from pydantic import ValidationError

from src import api
from src.api import (
    ChatCompletionRequest,
    InferenceBusy,
    InferenceUnavailable,
    app,
    cancel_inference,
    submit_inference,
)
from src.inference import ContextLimitError, GenerationCancelled, GenerationResult, MODEL_ID


def request_data(**changes):
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Hello"}],
        **changes,
    }


class RequestValidationTests(unittest.TestCase):
    def assert_rejected(self, data, code=None):
        # Exercise both Python callers and the JSON boundary used by HTTP clients.
        for source in (data, json.dumps(data)):
            with self.subTest(source_type=type(source).__name__):
                with self.assertRaises(ValidationError) as caught:
                    if isinstance(source, str):
                        ChatCompletionRequest.model_validate_json(source)
                    else:
                        ChatCompletionRequest.model_validate(source)
                if code is not None:
                    codes = {
                        error["type"]
                        for error in caught.exception.errors(include_input=False)
                    }
                    self.assertIn(code, codes)

    def test_valid_history_preserves_text_and_allows_consecutive_roles(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "  Hello\n世界!  "},
            {"role": "user", "content": "Another question."},
            {"role": "assistant", "content": "An earlier answer."},
        ]
        result = ChatCompletionRequest.model_validate_json(
            json.dumps(request_data(messages=messages))
        )
        self.assertEqual(
            [message.model_dump() for message in result.messages], messages
        )

    def test_valid_numeric_boundaries_and_integer_sampling_values(self):
        for settings in (
            {"max_tokens": 1, "temperature": 0, "top_p": 1},
            {"max_tokens": 512, "temperature": 2, "top_p": 1},
            {"temperature": 1, "top_p": 0.01},
            {"temperature": 0.7, "top_p": 0.9},
        ):
            with self.subTest(settings=settings):
                result = ChatCompletionRequest.model_validate_json(
                    json.dumps(request_data(**settings))
                )
                for field, value in settings.items():
                    self.assertEqual(getattr(result, field), value)

    def test_required_fields_and_model_identity(self):
        for field in ("model", "messages"):
            data = request_data()
            del data[field]
            self.assert_rejected(data, "missing")
        for model in ("another-model", "", 123, None):
            self.assert_rejected(request_data(model=model))

    def test_invalid_messages_and_content(self):
        for messages in (
            [],
            "Hello",
            None,
            [{"role": "tool", "content": "Hello"}],
            [{"role": "user"}],
            [{"content": "Hello"}],
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": " \t\n\u2003"}],
            [{"role": "user", "content": 123}],
            [{"role": "user", "content": [{"type": "image_url"}]}],
        ):
            with self.subTest(messages=messages):
                self.assert_rejected(request_data(messages=messages))

    def test_unknown_request_and_message_fields(self):
        for field in (
            "tools",
            "n",
            "top_k",
            "stop",
            "seed",
            "stream_options",
            "unknown",
        ):
            with self.subTest(field=field):
                self.assert_rejected(request_data(**{field: None}), "extra_forbidden")
        self.assert_rejected(
            request_data(messages=[{"role": "user", "content": "Hello", "name": "x"}]),
            "extra_forbidden",
        )

    def test_numeric_types_do_not_coerce_strings_or_booleans(self):
        for field in ("max_tokens", "temperature", "top_p"):
            for value in ("1", True, False, None):
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))
        self.assert_rejected(request_data(max_tokens=1.0), "int_type")

    def test_numeric_ranges(self):
        for field, values in (
            ("max_tokens", (0, -1, 513)),
            ("temperature", (-0.1, 2.01)),
            ("top_p", (0, -0.1, 1.01)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))

    def test_nonfinite_numbers(self):
        for field in ("max_tokens", "temperature", "top_p"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))

    def test_stream_requires_boolean(self):
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value):
                self.assert_rejected(request_data(stream=value), "bool_type")

    def test_greedy_top_p_combination_including_default_temperature(self):
        self.assert_rejected(request_data(top_p=0.9), "invalid_parameter_combination")
        self.assert_rejected(
            request_data(temperature=0, top_p=0.9), "invalid_parameter_combination"
        )

    def test_malformed_json_and_nonobject_body(self):
        for body in ('{"model":', "[]", "null", '"text"'):
            with self.subTest(body=body):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest.model_validate_json(body)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = object()
        loader_patch = patch("src.api.load_runtime", return_value=self.runtime)
        self.loader = loader_patch.start()
        self.addCleanup(loader_patch.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        self.addAsyncCleanup(self.client.aclose)

    async def test_startup_reuses_runtime_and_shutdown_releases_resources(self):
        async with app.router.lifespan_context(app):
            executor = app.state.executor
            for _ in range(2):
                health = await self.client.get("/health")
                ready = await self.client.get("/ready")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json(), {"status": "alive"})
                self.assertEqual(ready.status_code, 200)
                self.assertEqual(ready.json(), {
                    "ready": True, "status": "ready", "running": 0,
                    "queued": 0, "queue_capacity": 2,
                })
                self.assertIs(app.state.runtime, self.runtime)
            self.loader.assert_called_once_with()

        self.assertFalse(app.state.initialized)
        self.assertTrue(app.state.shutting_down)
        self.assertIsNone(app.state.runtime)
        self.assertIsNone(app.state.executor)
        with self.assertRaises(RuntimeError):
            executor.submit(lambda: None)

    async def test_loading_failure_prevents_serving(self):
        self.loader.side_effect = RuntimeError("Controlled loading failure")
        with self.assertRaises(RuntimeError):
            async with app.router.lifespan_context(app):
                self.fail("Startup failure must not enter the serving phase.")
        self.assertFalse(app.state.initialized)
        self.assertIsNone(app.state.runtime)
        self.assertIsNone(app.state.executor)
        ready = await self.client.get("/ready")
        self.assertEqual(ready.status_code, 503)
        self.assertFalse(ready.json()["ready"])

    async def test_executor_creation_failure_releases_loaded_runtime(self):
        with patch("src.api.ThreadPoolExecutor", side_effect=RuntimeError("No worker")):
            with self.assertRaises(RuntimeError):
                async with app.router.lifespan_context(app):
                    self.fail("Partial startup must not enter the serving phase.")
        self.assertFalse(app.state.initialized)
        self.assertIsNone(app.state.runtime)
        self.assertIsNone(app.state.executor)

    async def test_health_remains_available_when_inference_slot_is_busy(self):
        started, release = Event(), Event()

        def held(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Producer was not released")
            return GenerationResult("ok", 1, [2], "stop")

        async with app.router.lifespan_context(app):
            with patch("src.api.generate", side_effect=held):
                job = submit_inference(app, ChatCompletionRequest.model_validate(request_data()))
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    self.assertEqual((await self.client.get("/health")).status_code, 200)
                    ready = await self.client.get("/ready")
                    self.assertEqual(ready.status_code, 200)
                    self.assertEqual(ready.json(), {
                        "ready": True, "status": "busy", "running": 1,
                        "queued": 0, "queue_capacity": 2,
                    })
                finally:
                    release.set()
                    await job.result
            self.assertTrue((await self.client.get("/ready")).json()["ready"])

    async def test_uninitialized_application_does_not_claim_readiness(self):
        with (
            patch.object(app.state, "initialized", False, create=True),
            patch.object(app.state, "shutting_down", False, create=True),
        ):
            ready = await self.client.get("/ready")
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(ready.json()["status"], "not_ready")

    async def test_shutdown_waits_for_worker_without_blocking_http_loop(self):
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        started = Event()
        release = Event()
        shutdown_started = Event()
        executor = app.state.executor

        def held_producer(*args, **kwargs):
            started.set()
            release.wait()
            return app.state.runtime

        generation_patch = patch("src.api.generate", side_effect=held_producer)
        generation_patch.start()
        self.addCleanup(generation_patch.stop)
        job = submit_inference(app, ChatCompletionRequest.model_validate(request_data()))
        prepared = asyncio.get_running_loop().create_future()
        queued = submit_inference(
            app, ChatCompletionRequest.model_validate(request_data()), prepared=prepared,
        )
        original_shutdown = executor.shutdown

        def observed_shutdown(*args, **kwargs):
            shutdown_started.set()
            return original_shutdown(*args, **kwargs)

        # Release even if a regression blocks the event loop during shutdown.
        safety_release = Timer(5, release.set)
        safety_release.start()
        shutdown_task = None
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            with patch.object(executor, "shutdown", side_effect=observed_shutdown):
                shutdown_task = asyncio.create_task(context.__aexit__(None, None, None))
                self.assertTrue(await asyncio.to_thread(shutdown_started.wait, 2))
                self.assertFalse(shutdown_task.done())
                self.assertFalse(job.result.done())
                self.assertEqual(job.cancellation.reason, "shutdown")
                self.assertEqual(queued.state, "canceled")
                for future in (queued.result, prepared):
                    with self.assertRaises(GenerationCancelled) as canceled:
                        await future
                    self.assertEqual(canceled.exception.reason, "shutdown")
                self.assertIsNone(queued.producer)
                self.assertIs(app.state.runtime, self.runtime)
                self.assertFalse(app.state.initialized)
                health = await self.client.get("/health")
                ready = await self.client.get("/ready")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(ready.status_code, 503)
                self.assertEqual(
                    ready.json()["status"], "shutting_down"
                )
                with self.assertRaises(InferenceUnavailable):
                    submit_inference(
                        app, ChatCompletionRequest.model_validate(request_data())
                    )
                release.set()
                await asyncio.wait_for(shutdown_task, timeout=3)
            with self.assertRaises(GenerationCancelled):
                await job.result
            self.assertIsNone(app.state.runtime)
            self.assertIsNone(app.state.executor)
            self.assertEqual(app.state.producer_futures, set())
            self.assertIsNone(app.state.active_job)
        finally:
            release.set()
            safety_release.cancel()
            safety_release.join()
            if shutdown_task is None:
                await context.__aexit__(None, None, None)
            else:
                await shutdown_task


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        loader_patch = patch("src.api.load_runtime", return_value=object())
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        self.addAsyncCleanup(context.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.request = ChatCompletionRequest.model_validate(request_data())
        self.result = GenerationResult("answer", 3, [7], "length")

    async def test_unavailable_state_rejects_before_submitting_work(self):
        for initialized, shutting_down in ((False, False), (True, True)):
            with self.subTest(initialized=initialized, shutting_down=shutting_down):
                with (
                    patch.object(app.state, "initialized", initialized),
                    patch.object(app.state, "shutting_down", shutting_down),
                ):
                    with self.assertRaises(InferenceUnavailable):
                        submit_inference(app, self.request)
                self.assertIsNone(app.state.active_job)
                self.assertEqual(app.state.producer_futures, set())

    async def test_fifo_capacity_and_cancellation_retain_actual_worker_ownership(self):
        started, release = Event(), Event()
        loop_thread = get_ident()
        calls = []

        def held_generation(runtime, messages, **kwargs):
            self.assertNotEqual(get_ident(), loop_thread)
            name = messages[0]["content"]
            calls.append(name)
            if name == "first":
                started.set()
                if not release.wait(5):
                    raise TimeoutError("Test producer was not released.")
            return self.result

        def admit(name):
            return submit_inference(app, ChatCompletionRequest.model_validate(
                request_data(messages=[{"role": "user", "content": name}]),
            ))

        with patch("src.api.generate", side_effect=held_generation):
            first = admit("first")
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                second, third = admit("second"), admit("third")
                self.assertIsNone(second.producer)
                self.assertIsNone(third.producer)
                self.assertEqual(calls, ["first"])
                self.assertEqual(app.state.producer_futures, {first.producer})
                with self.assertRaises(InferenceBusy):
                    admit("overflow")
                self.assertEqual((await self.client.get("/health")).status_code, 200)
                ready = await self.client.get("/ready")
                self.assertEqual(ready.status_code, 503)
                self.assertEqual(ready.json(), {
                    "ready": False, "status": "full", "running": 1,
                    "queued": 2, "queue_capacity": 2,
                })

                cancel_inference(app, second)
                with self.assertRaises(GenerationCancelled):
                    await second.result
                self.assertIsNone(second.started_at)
                self.assertEqual((await self.client.get("/ready")).status_code, 200)
                fourth = admit("fourth")
                cancel_inference(app, first)
                self.assertIs(app.state.active_job, first)
                self.assertFalse(first.producer.done())
                self.assertEqual(calls, ["first"])
                with self.assertRaises(InferenceBusy):
                    admit("still full")
            finally:
                release.set()
                await asyncio.gather(first.result, return_exceptions=True)
            self.assertEqual(first.state, "canceled")
            self.assertEqual(await third.result, self.result)
            self.assertEqual(await fourth.result, self.result)
            self.assertEqual(calls, ["first", "third", "fourth"])
            self.assertIsNone(app.state.active_job)
            self.assertEqual(app.state.producer_futures, set())
            self.assertEqual((await self.client.get("/ready")).status_code, 200)

    async def test_context_and_generation_failures_release_slot_for_recovery(self):
        for failure in (ContextLimitError("Oversized request"), RuntimeError("Failed")):
            with self.subTest(failure=type(failure).__name__):
                with patch("src.api.generate", side_effect=[failure, self.result]):
                    with self.assertRaises(type(failure)):
                        await submit_inference(app, self.request).result
                    self.assertIsNone(app.state.active_job)
                    self.assertEqual(app.state.producer_futures, set())
                    recovered = await submit_inference(app, self.request).result
                    self.assertEqual(recovered, self.result)

    async def test_submission_failure_releases_slot_for_recovery(self):
        with patch.object(app.state.executor, "submit", side_effect=RuntimeError("No worker")):
            with self.assertRaises(RuntimeError):
                await submit_inference(app, self.request).result
        self.assertIsNone(app.state.active_job)
        self.assertEqual(app.state.producer_futures, set())
        with patch("src.api.generate", return_value=self.result):
            recovered = await submit_inference(app, self.request).result
        self.assertEqual(recovered, self.result)

    async def test_worker_failures_promote_fifo_and_wake_preparation(self):
        for failure in (ContextLimitError("Token limit"), RuntimeError("Private failure")):
            with self.subTest(failure=type(failure).__name__):
                started, release = Event(), Event()
                calls = []

                def generate(*args, **kwargs):
                    calls.append(len(calls))
                    if len(calls) == 1:
                        started.set()
                        if not release.wait(5):
                            raise TimeoutError("Producer was not released")
                        raise failure
                    return self.result

                with patch("src.api.generate", side_effect=generate):
                    prepared = asyncio.get_running_loop().create_future()
                    first = submit_inference(app, self.request, prepared=prepared)
                    queued = []
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 2))
                        queued = [submit_inference(app, self.request) for _ in range(2)]
                        self.assertTrue(all(job.producer is None for job in queued))
                    finally:
                        release.set()
                    for future in (first.result, prepared):
                        with self.assertRaises(type(failure)):
                            await future
                    self.assertEqual(await queued[0].result, self.result)
                    self.assertEqual(await queued[1].result, self.result)
                    self.assertLessEqual(queued[0].finished_at, queued[1].started_at)
                    self.assertEqual(calls, [0, 1, 2])
                    self.assertIsNone(app.state.active_job)

    async def test_failed_submission_during_promotion_does_not_strand_next_job(self):
        started, release = Event(), Event()

        def held(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Producer was not released")
            return self.result

        original_submit = app.state.executor.submit
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("Worker submission failed")
            return original_submit(*args, **kwargs)

        with patch("src.api.generate", side_effect=held):
            first = submit_inference(app, self.request)
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                prepared = asyncio.get_running_loop().create_future()
                second = submit_inference(app, self.request, prepared=prepared)
                third = submit_inference(app, self.request)
                with patch.object(app.state.executor, "submit", side_effect=fail_once):
                    release.set()
                    await first.result
                    for future in (second.result, prepared):
                        with self.assertRaises(RuntimeError):
                            await future
                    self.assertEqual(await third.result, self.result)
                self.assertIsNone(second.started_at)
                self.assertEqual(second.state, "failed")
                self.assertIsNone(app.state.active_job)
            finally:
                release.set()
                await asyncio.gather(first.result, return_exceptions=True)

    async def test_slot_is_released_by_raw_future_completion_not_worker_finally(self):
        finished_generation, release_future = Event(), Event()
        original_run = api._run_inference
        first_job = None

        def held_exit(runtime, job, loop):
            result = original_run(runtime, job, loop)
            if job is first_job:
                finished_generation.set()
                if not release_future.wait(5):
                    raise TimeoutError("Raw worker was not released")
            return result

        # Hold generation until first_job is assigned, then hold the raw worker exit.
        begin = Event()

        def generate(*args, **kwargs):
            if not begin.wait(5):
                raise TimeoutError("Generation was not released")
            return self.result

        with (
            patch("src.api._run_inference", side_effect=held_exit),
            patch("src.api.generate", side_effect=generate) as generation,
        ):
            first_job = submit_inference(app, self.request)
            begin.set()
            try:
                self.assertTrue(await asyncio.to_thread(finished_generation.wait, 2))
                second = submit_inference(app, self.request)
                self.assertIs(app.state.active_job, first_job)
                self.assertIsNotNone(first_job.finished_at)
                self.assertFalse(first_job.producer.done())
                self.assertFalse(first_job.result.done())
                self.assertIsNone(second.producer)
                self.assertEqual(generation.call_count, 1)
            finally:
                release_future.set()
                await first_job.result
            self.assertEqual(await second.result, self.result)

    async def test_delivery_abort_after_worker_exit_does_not_relabel_success(self):
        original_done = api._producer_done

        def abort_before_callback(application, job, producer):
            self.assertTrue(producer.done())
            self.assertIsNotNone(job.finished_at)
            cancel_inference(application, job, "slow_consumer")
            original_done(application, job, producer)

        with (
            patch("src.api.generate", return_value=self.result),
            patch("src.api._producer_done", side_effect=abort_before_callback),
        ):
            job = submit_inference(app, self.request)
            self.assertEqual(await job.result, self.result)
        self.assertEqual(job.state, "completed")
        self.assertFalse(job.cancellation.is_cancelled())
        self.assertIsNone(app.state.active_job)

    async def test_worker_exit_between_cancellation_check_and_signal_preserves_success(self):
        entered, release, exited = Event(), Event(), Event()
        original_worker = api._run_inference

        def held_result(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Generation was not released")
            return self.result

        def observed_exit(*args):
            try:
                return original_worker(*args)
            finally:
                exited.set()

        with (
            patch("src.api.generate", side_effect=held_result),
            patch("src.api._run_inference", side_effect=observed_exit),
            self.assertLogs(api.logger, level="INFO") as captured,
        ):
            job = submit_inference(
                app, self.request, request_log=api._RequestLog(api.perf_counter()),
            )
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                original_cancel = job.cancellation.cancel

                def signal_after_exit(reason):
                    release.set()
                    if not exited.wait(2):
                        raise TimeoutError("Worker did not exit")
                    original_cancel(reason)

                with patch.object(job.cancellation, "cancel", side_effect=signal_after_exit):
                    cancel_inference(app, job)
                self.assertLess(job.finished_at, job.cancellation.requested_at)
                self.assertEqual(await job.result, self.result)
            finally:
                release.set()
        self.assertEqual(job.state, "completed")
        self.assertIsNone(app.state.active_job)
        records = [json.loads(record.getMessage()) for record in captured.records]
        for event in ("producer_finished", "scheduling_finished"):
            record = next(row for row in records if row["event"] == event)
            self.assertIsNone(record["cancellation_delay_ms"])
            self.assertNotIn("cancellation_reason", record)
        producer = next(row for row in records if row["event"] == "producer_finished")
        self.assertEqual(producer["outcome"], "completed")
        self.assertEqual(producer["completion_tokens"], len(self.result.generated_ids))

    async def test_abandoned_failure_is_observed_and_does_not_leak_busy_slot(self):
        started = Event()
        release = Event()
        completed = asyncio.Event()
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))

        def failing_generation(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test producer was not released.")
            raise RuntimeError("Private failure details")

        try:
            with patch("src.api.generate", side_effect=failing_generation):
                job = submit_inference(app, self.request)
                producer = next(iter(app.state.producer_futures))
                producer.add_done_callback(lambda future: completed.set())
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    cancel_inference(app, job)
                    self.assertIs(app.state.active_job, job)
                    self.assertFalse(job.producer.done())
                finally:
                    release.set()
                    await asyncio.wait_for(completed.wait(), timeout=3)
                self.assertIsNone(app.state.active_job)
                self.assertEqual(app.state.producer_futures, set())
                # Never await or retrieve the producer exception in this test.
                del producer
                del job
            await asyncio.sleep(0)
            gc.collect()
            self.assertEqual(unhandled, [])
            with patch("src.api.generate", return_value=self.result):
                recovered = await submit_inference(app, self.request).result
            self.assertEqual(recovered, self.result)
        finally:
            release.set()
            loop.set_exception_handler(previous_handler)


class CompletionEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        loader_patch = patch("src.api.load_runtime", return_value=object())
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        self.addAsyncCleanup(context.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.path = "/v1/chat/completions"
        self.result = GenerationResult("answer", 3, [7, 2], "stop")

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status)
        error = response.json()["error"]
        self.assertEqual(error["code"], code)
        request_id = response.headers["X-Request-ID"]
        self.assertEqual(UUID(request_id).hex, request_id)
        self.assertEqual(error["request_id"], request_id)

    async def test_validation_errors_are_private_and_precede_busy_admission(self):
        marker = "PRIVATE-INPUT-MARKER"
        cases = [
            (b'{"messages": "PRIVATE-INPUT-MARKER",', "invalid_request"),
            (b"\xff", "invalid_request"),
            (b"[]", "invalid_request"),
            ({}, "invalid_request"),
            (request_data(model=marker), "invalid_request"),
            (request_data(messages=[]), "invalid_request"),
            (request_data(messages=[{"role": marker, "content": marker}]), "invalid_request"),
            (request_data(messages=[{"role": "user", "content": " \t\n"}]), "invalid_request"),
            (request_data(max_tokens=-1), "invalid_request"),
            (request_data(max_tokens=marker), "invalid_request"),
            (request_data(temperature=True), "invalid_request"),
            (request_data(temperature=float("nan")), "invalid_request"),
            (request_data(**{marker: marker}), "invalid_request"),
            (request_data(messages=[{"role": "user", "content": marker, marker: marker}]), "invalid_request"),
            (request_data(top_p=0.9), "invalid_parameter_combination"),
        ]
        with (
            patch.object(app.state, "active_job", object()),
            patch.object(app.state, "pending_jobs", [object(), object()]),
        ):
            for body, code in cases:
                with self.subTest(body=body):
                    response = await self.client.post(
                        self.path,
                        content=body if isinstance(body, bytes) else json.dumps(body),
                        headers={
                            "Content-Type": "application/json",
                            "X-Request-ID": marker,
                        },
                    )
                    self.assert_error(response, 422, code)
                    self.assertNotIn(marker, response.text)
                    self.assertNotEqual(response.headers["X-Request-ID"], marker)
            self.assertEqual(app.state.producer_futures, set())

    async def test_unavailable_application_returns_503(self):
        for initialized, shutting_down in ((False, False), (True, True)):
            with self.subTest(initialized=initialized, shutting_down=shutting_down):
                with (
                    patch.object(app.state, "initialized", initialized),
                    patch.object(app.state, "shutting_down", shutting_down),
                ):
                    response = await self.client.post(self.path, json=request_data())
                self.assert_error(response, 503, "not_ready")
                self.assertIsNone(app.state.active_job)

    async def test_preparation_and_generation_errors_allow_later_success(self):
        marker = "PRIVATE-INPUT-MARKER"
        for failure, status, code in (
            (
                ContextLimitError(
                    "Prompt tokens (2000) plus requested output tokens (100) "
                    "exceed the supported limit (2048)."
                ),
                400,
                "context_length_exceeded",
            ),
            (RuntimeError(marker), 500, "generation_failed"),
        ):
            with self.subTest(code=code):
                with patch("src.api.generate", side_effect=[failure, self.result]):
                    response = await self.client.post(
                        self.path,
                        json=request_data(messages=[{"role": "user", "content": marker}]),
                    )
                    self.assert_error(response, status, code)
                    self.assertNotIn(marker, response.text)
                    self.assertIsNone(app.state.active_job)
                    recovered = await self.client.post(self.path, json=request_data())
                    self.assertEqual(recovered.status_code, 200)
                ready = await self.client.get("/ready")
                self.assertEqual(ready.status_code, 200)

    async def test_submission_failure_returns_safe_500_and_releases_slot(self):
        with patch.object(app.state.executor, "submit", side_effect=RuntimeError("SECRET")):
            response = await self.client.post(self.path, json=request_data())
        self.assert_error(response, 500, "generation_failed")
        self.assertNotIn("SECRET", response.text)
        self.assertIsNone(app.state.active_job)
        with patch("src.api.generate", return_value=self.result):
            recovered = await self.client.post(self.path, json=request_data())
        self.assertEqual(recovered.status_code, 200)

    async def test_full_queue_and_canceled_http_handler_retain_producer_ownership(self):
        started, release = Event(), Event()

        def held_generation(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test producer was not released.")
            return self.result

        with patch("src.api.generate", side_effect=held_generation):
            first = asyncio.create_task(self.client.post(self.path, json=request_data()))
            queued = []
            producer = None
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                active = app.state.active_job
                producer = active.producer
                queued = [
                    submit_inference(app, ChatCompletionRequest.model_validate(request_data()))
                    for _ in range(2)
                ]
                self.assertEqual((await self.client.get("/health")).status_code, 200)
                self.assertEqual((await self.client.get("/ready")).status_code, 503)
                overflow = await self.client.post(self.path, json=request_data())
                self.assert_error(overflow, 429, "server_busy")
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertFalse(producer.done())
                self.assertIs(app.state.active_job, active)
                self.assertEqual(active.cancellation.reason, "request_canceled")
                overflow = await self.client.post(self.path, json=request_data())
                self.assert_error(overflow, 429, "server_busy")
            finally:
                release.set()
                if producer is not None:
                    await asyncio.gather(producer, return_exceptions=True)
                await asyncio.gather(first, *(job.result for job in queued), return_exceptions=True)
            recovered = await self.client.post(self.path, json=request_data())
            self.assertEqual(recovered.status_code, 200)
            self.assertIsNone(app.state.active_job)

    async def test_unknown_routes_and_wrong_methods_keep_normal_routing(self):
        missing = await self.client.post("/not-a-route", json=request_data())
        wrong_method = await self.client.get(self.path)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(wrong_method.status_code, 405)
        self.assertIn("POST", wrong_method.headers["allow"])
        self.assertNotIn("error", missing.json())
        self.assertNotIn("error", wrong_method.json())


if __name__ == "__main__":
    unittest.main()
