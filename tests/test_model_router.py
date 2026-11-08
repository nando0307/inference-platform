import asyncio
import json
import unittest
from contextlib import suppress
from unittest.mock import patch

import httpx
from pydantic import ValidationError

from benchmarks.client import run_request
from src import model_router as router
from src.vllm_engine import HIGH_MODEL, LOW_MODEL


def data(model=LOW_MODEL, **changes):
    return {
        "model": model, "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1, **changes,
    }


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


class FakeEngine:
    def __init__(self):
        self.model = None
        self.alive = False
        self.base_url = "http://127.0.0.1:1"
        self.calls = []
        self.start_gate = asyncio.Event()
        self.start_gate.set()
        self.fail_load = False
        self.fail_stop = False
        self.manager = None

    def is_alive(self):
        return self.alive

    async def prepare(self, model, messages, max_tokens):
        if messages[0]["content"] == "over context":
            raise router.ContextLimitError("private prompt")
        if messages[0]["content"] == "invalid checkpoint":
            raise ValueError("private checkpoint configuration")
        return {"prompt_tokens": 3, "prompt_token_ids": [10, 11, 12],
                "eos_token_ids": [2], "chat_template_sha256": "template",
                "formatted_prompt_sha256": "prompt"}

    async def start(self, model):
        if self.model is not None:
            raise AssertionError("Starting over a resident engine")
        self.calls.append(("start", model))
        await self.start_gate.wait()
        if self.fail_load:
            raise RuntimeError("private loading failure")
        self.model, self.alive = model, True

    async def stop(self):
        if self.manager is not None and self.manager.active:
            raise AssertionError("Unloading an active engine")
        if self.model is not None:
            self.calls.append(("stop", self.model))
        if self.fail_stop:
            raise RuntimeError("private cleanup failure")
        self.model, self.alive = None, False


class ValidationTests(unittest.TestCase):
    def test_exact_allowlist_and_strict_public_boundary(self):
        for model in (LOW_MODEL, HIGH_MODEL):
            self.assertEqual(router.RoutingRequest.model_validate(data(model)).model, model)
        invalid = [
            data("Qwen/other"), data(max_tokens=True), data(max_tokens=513),
            data(stream="true"), data(temperature=True), data(temperature=float("nan")),
            data(top_p=0.9), data(messages=[]), data(messages=[{"role": "tool", "content": "x"}]),
            data(messages=[{"role": "user", "content": [{"type": "image_url"}]}]),
            data(messages=[{"role": "user", "content": "x"}] * 65),
            data(messages=[{"role": "user", "content": "x" * (router.MAX_PROMPT_CHARS + 1)}]),
            data(chat_template_kwargs={"enable_thinking": True}), data(ignore_eos=True),
            data(return_token_ids=True), data(stream_options={"include_usage": False}),
            data(base_url="http://private"), data(messages=[{"role": "user", "content": " "}]),
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                router.RoutingRequest.model_validate(payload)


class ManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = FakeEngine()
        self.manager = router.ModelManager(self.engine)
        self.engine.manager = self.manager
        self.manager.start()
        self.addAsyncCleanup(self.cleanup)

    async def cleanup(self):
        for lease in list(self.manager.active):
            await self.manager.release(lease, True)
        await self.manager.shutdown()

    def submit(self, model=LOW_MODEL, name="request", **changes):
        return self.manager.submit(router.RoutingRequest.model_validate(data(model, **changes)), name)

    async def acquire(self, model=LOW_MODEL, name="request"):
        lease = self.submit(model, name)
        return await asyncio.wait_for(asyncio.shield(lease.ready), 2)

    async def test_cold_concurrent_requests_share_one_load(self):
        self.engine.start_gate.clear()
        first = self.submit(name="first")
        second = self.submit(name="second")
        await until(lambda: self.manager.state == "loading")
        snapshot = self.manager.snapshot()
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["queued"], 2)
        self.assertFalse(snapshot["accepting"])
        self.engine.start_gate.set()
        await asyncio.gather(first.ready, second.ready)
        self.assertEqual(self.engine.calls, [("start", LOW_MODEL)])
        self.assertEqual(self.manager.snapshot()["active"], 2)

    async def test_cross_model_fifo_blocks_same_model_overtaking(self):
        first = await self.acquire(name="low-a")
        second = await self.acquire(name="low-b")
        high = self.submit(HIGH_MODEL, "high")
        low = self.submit(name="later-low")
        await until(lambda: self.manager.state == "draining")
        await self.manager.release(first, True)
        await asyncio.sleep(0)
        self.assertFalse(high.ready.done())
        self.assertFalse(low.ready.done())
        self.assertNotIn(("stop", LOW_MODEL), self.engine.calls)
        await self.manager.release(second, True)
        await asyncio.wait_for(high.ready, 2)
        self.assertFalse(low.ready.done())
        await self.manager.release(high, True)
        await asyncio.wait_for(low.ready, 2)
        self.assertEqual(self.engine.calls, [
            ("start", LOW_MODEL), ("stop", LOW_MODEL),
            ("start", HIGH_MODEL), ("stop", HIGH_MODEL),
            ("start", LOW_MODEL),
        ])
        admitted = [event["request_id"] for event in self.manager.events if event["event"] == "admitted"]
        self.assertEqual(admitted, ["low-a", "low-b", "high", "later-low"])

    async def test_capacity_and_waiting_cancellation_remove_fifo_head(self):
        first = await self.acquire(name="a")
        second = await self.acquire(name="b")
        canceled = self.submit(HIGH_MODEL, "cancel")
        survivor = self.submit(name="survivor")
        with self.assertRaises(router.RouterError) as caught:
            self.submit(name="overflow")
        self.assertEqual(caught.exception.status, 429)
        self.manager.cancel_waiting(canceled)
        self.assertTrue(canceled.ready.cancelled())
        await self.manager.release(first, True)
        await asyncio.wait_for(survivor.ready, 2)
        self.assertEqual(self.engine.calls, [("start", LOW_MODEL)])
        self.assertIn(second, self.manager.active)
        self.assertTrue(any(event["event"] == "canceled" and event["request_id"] == "cancel"
                            for event in self.manager.events))

    async def test_active_cancellation_drains_before_process_recycle(self):
        first = await self.acquire(name="cancel")
        second = await self.acquire(name="remaining")
        first.dispatched = True
        following = self.submit(name="following")
        await self.manager.release(first, False)
        await until(lambda: self.manager.state == "draining")
        self.assertFalse(following.ready.done())
        self.assertEqual(self.engine.calls, [("start", LOW_MODEL)])
        await self.manager.release(second, True)
        await asyncio.wait_for(following.ready, 2)
        self.assertEqual(self.engine.calls, [("start", LOW_MODEL),
                                            ("stop", LOW_MODEL), ("start", LOW_MODEL)])

    async def test_context_rejection_precedes_loading_and_does_not_poison_queue(self):
        invalid = self.submit(messages=[{"role": "user", "content": "over context"}])
        with self.assertRaises(router.RouterError) as caught:
            await invalid.ready
        self.assertEqual(caught.exception.code, "context_length_exceeded")
        self.assertEqual(self.engine.calls, [])
        valid = await self.acquire()
        self.assertEqual(valid.prepared["prompt_token_ids"], [10, 11, 12])

    async def test_invalid_checkpoint_preparation_fails_closed_not_as_client_context(self):
        invalid = self.submit(messages=[{"role": "user", "content": "invalid checkpoint"}])
        with self.assertRaises(router.RouterError) as caught:
            await invalid.ready
        self.assertEqual(caught.exception.code, "engine_failed")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(self.engine.calls, [])
        self.assertFalse(self.manager.snapshot()["accepting"])

    async def test_failed_load_fails_all_waiters_without_retry(self):
        self.engine.fail_load = True
        first, second = self.submit(name="a"), self.submit(name="b")
        results = await asyncio.gather(first.ready, second.ready, return_exceptions=True)
        self.assertTrue(all(isinstance(result, router.RouterError) for result in results))
        self.assertEqual(self.manager.state, "failed")
        with self.assertRaises(router.RouterError):
            self.submit()
        self.assertEqual(self.engine.calls, [("start", LOW_MODEL)])

    async def test_dead_engine_fails_queued_work_and_never_switches(self):
        active = await self.acquire()
        waiting = self.submit(HIGH_MODEL)
        self.engine.alive = False
        self.manager.wake.set()
        with self.assertRaises(router.RouterError):
            await asyncio.wait_for(waiting.ready, 2)
        self.assertFalse(self.manager.snapshot()["accepting"])
        await self.manager.release(active, False)
        await asyncio.wait_for(self.manager.pump, 2)
        self.assertEqual(self.engine.calls, [("start", LOW_MODEL), ("stop", LOW_MODEL)])

    async def test_failed_cleanup_never_loads_another_model_and_is_reported(self):
        engine = FakeEngine()
        manager = router.ModelManager(engine)
        engine.manager = manager
        manager.start()
        first = manager.submit(router.RoutingRequest.model_validate(data()), "low")
        await first.ready
        next_model = manager.submit(router.RoutingRequest.model_validate(data(HIGH_MODEL)), "high")
        engine.fail_stop = True
        await manager.release(first, True)
        with self.assertRaises(router.RouterError):
            await asyncio.wait_for(next_model.ready, 2)
        await asyncio.wait_for(manager.pump, 2)
        self.assertNotIn(("start", HIGH_MODEL), engine.calls)
        self.assertFalse(manager.snapshot()["accepting"])
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            await manager.shutdown()

    async def test_shutdown_cancels_owned_http_task_before_stopping(self):
        lease = await self.acquire()
        released = asyncio.Event()

        async def hold():
            try:
                await asyncio.Event().wait()
            finally:
                await self.manager.release(lease, False)
                released.set()

        lease.task = asyncio.create_task(hold())
        await asyncio.sleep(0)
        with patch.object(router, "DRAIN_TIMEOUT", 0.02):
            await asyncio.wait_for(self.manager.shutdown(), 1)
        self.assertTrue(released.is_set())
        self.assertIsNone(self.engine.model)
        self.assertFalse(self.manager.snapshot()["accepting"])
        with suppress(asyncio.CancelledError):
            await lease.task


class LocalUpstream:
    """Real loopback HTTP, including observable socket closure on cancellation."""

    def __init__(self):
        self.requests = []
        self.tasks = set()
        self.first_text = asyncio.Event()
        self.disconnected = asyncio.Event()
        self.hold_stream = False
        self.wrong_model = False
        self.wrong_id = False
        self.missing_prompt_ids = False
        self.truncated = False
        self.reject = False

    async def start(self):
        self.server = await asyncio.start_server(self.handle, "127.0.0.1", 0)
        return "http://127.0.0.1:" + str(self.server.sockets[0].getsockname()[1])

    async def close(self):
        self.server.close()
        await self.server.wait_closed()
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def handle(self, reader, writer):
        self.tasks.add(asyncio.current_task())
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            length = next(int(line.split(b":", 1)[1]) for line in headers.splitlines()
                          if line.lower().startswith(b"content-length:"))
            request_id = next(line.split(b":", 1)[1].strip().decode() for line in headers.splitlines()
                              if line.lower().startswith(b"x-request-id:"))
            body = json.loads(await reader.readexactly(length))
            self.requests.append(body)
            if self.reject:
                writer.write(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 18\r\nConnection: close\r\n\r\nPRIVATE-ERROR-TEXT")
                await writer.drain()
                return
            model = HIGH_MODEL if self.wrong_model else body["model"]
            upstream_id = "wrong-id" if self.wrong_id else "chatcmpl-" + request_id
            base = {"id": upstream_id, "object": "chat.completion.chunk", "created": 1, "model": model}
            prompt = {} if self.missing_prompt_ids else {"prompt_token_ids": [10, 11, 12]}
            usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            if not body["stream"]:
                payload = {**base, **prompt, "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"},
                                                         "finish_reason": "length", "token_ids": [13]}], "usage": usage}
                raw = json.dumps(payload).encode()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: "
                             + str(len(raw)).encode() + b"\r\n\r\n" + raw)
                await writer.drain()
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            first = {**base, **prompt, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"},
                                                    "finish_reason": None, "token_ids": [13]}]}
            writer.write(router._sse(first).encode())
            await writer.drain()
            self.first_text.set()
            if self.hold_stream:
                await reader.read()
                self.disconnected.set()
                return
            if not self.truncated:
                writer.write(router._sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}).encode())
                writer.write(router._sse({**base, "choices": [], "usage": usage}).encode())
                writer.write(b"data: [DONE]\n\n")
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            self.disconnected.set()
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()
            self.tasks.discard(asyncio.current_task())


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.upstream = LocalUpstream()
        self.engine = FakeEngine()
        self.engine.base_url = await self.upstream.start()
        self.addAsyncCleanup(self.upstream.close)
        self.app = router.create_app(self.engine)
        context = self.app.router.lifespan_context(self.app)
        await context.__aenter__()
        self.addAsyncCleanup(context.__aexit__, None, None, None)
        self.engine.manager = self.app.state.manager
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://router")
        self.addAsyncCleanup(self.client.aclose)

    async def test_nonstream_routes_exact_identity_and_returns_usage(self):
        cold = await self.client.get("/ready")
        self.assertEqual(cold.status_code, 503)
        self.assertEqual(cold.json()["state"], "unloaded")
        self.assertTrue(cold.json()["accepting"])
        for model in (LOW_MODEL, HIGH_MODEL):
            response = await self.client.post("/v1/chat/completions", json=data(model))
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result["model"], model)
            self.assertEqual(result["id"], "chatcmpl-" + response.headers["x-request-id"])
            self.assertEqual(result["choices"][0]["message"]["content"], "hello")
            self.assertEqual(result["prompt_token_ids"], [10, 11, 12])
            self.assertEqual(result["choices"][0]["token_ids"], [13])
            self.assertEqual(result["usage"], {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})
        self.assertEqual(self.app.state.manager.snapshot()["active"], 0)
        self.assertTrue((await self.client.get("/ready")).json()["ready"])

    async def test_stream_satisfies_existing_strict_client(self):
        workload = {"model": HIGH_MODEL, "generation": {"temperature": 0, "top_p": 1}}
        case = {"name": "identity", "messages": data()["messages"], "max_tokens": 1, "prompt_tokens": 3}
        row = await run_request(self.client, "http://router", workload, case, "measured", 0, capture_events=True)
        self.assertEqual(row["status"], "success", row)
        self.assertEqual(row["text"], "hello")
        self.assertEqual(row["prompt_token_ids"], [10, 11, 12])
        self.assertEqual(row["events"][0]["token_ids"], [13])
        self.assertEqual(row["finish_reason"], "length")
        self.assertEqual(self.app.state.manager.snapshot()["active"], 0)

    async def test_invalid_requests_reject_before_any_load_and_hide_inputs(self):
        for payload in (data("PRIVATE-MODEL"), data(backend="PRIVATE-BACKEND"), data(max_tokens=513)):
            response = await self.client.post("/v1/chat/completions", json=payload)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn("PRIVATE", response.text)
        response = await self.client.post("/v1/chat/completions", content=b"x" * (router.MAX_BODY_BYTES + 1))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.engine.calls, [])
        self.assertEqual(self.upstream.requests, [])

    async def test_wrong_upstream_model_never_reaches_client_as_a_completion(self):
        self.upstream.wrong_model = True
        response = await self.client.post("/v1/chat/completions", json=data())
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("hello", response.text)
        self.assertNotIn("choices", response.json())
        self.assertFalse(self.app.state.manager.snapshot()["accepting"])
        again = await self.client.post("/v1/chat/completions", json=data())
        self.assertEqual(again.status_code, 503)
        self.assertEqual(len(self.upstream.requests), 1)

    async def assert_invalid_upstream_identity(self, flag, stream):
        setattr(self.upstream, flag, True)
        response = await self.client.post("/v1/chat/completions", json=data(stream=stream))
        self.assertEqual(response.status_code, 200 if stream else 502)
        self.assertIn("upstream_failed", response.text)
        self.assertNotIn("hello", response.text)
        self.assertNotIn('"choices"', response.text)
        self.assertFalse(self.app.state.manager.snapshot()["accepting"])

    async def test_nonstream_wrong_request_id_is_rejected(self):
        await self.assert_invalid_upstream_identity("wrong_id", False)

    async def test_stream_wrong_request_id_is_rejected(self):
        await self.assert_invalid_upstream_identity("wrong_id", True)

    async def test_nonstream_missing_prompt_token_ids_is_rejected(self):
        await self.assert_invalid_upstream_identity("missing_prompt_ids", False)

    async def test_stream_missing_prompt_token_ids_is_rejected(self):
        await self.assert_invalid_upstream_identity("missing_prompt_ids", True)

    async def test_upstream_errors_are_sanitized_and_not_retried(self):
        self.upstream.reject = True
        response = await self.client.post("/v1/chat/completions", json=data())
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("PRIVATE", response.text)
        self.assertEqual(len(self.upstream.requests), 1)
        self.assertFalse(self.app.state.manager.snapshot()["accepting"])

    async def test_truncated_stream_has_error_not_success(self):
        self.upstream.truncated = True
        workload = {"model": LOW_MODEL, "generation": {"temperature": 0, "top_p": 1}}
        case = {"name": "truncated", "messages": data()["messages"], "max_tokens": 1, "prompt_tokens": 3}
        row = await run_request(self.client, "http://router", workload, case, "measured", 0)
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_code"], "upstream_failed")
        self.assertEqual(self.app.state.manager.snapshot()["active"], 0)

    async def asgi_stream(self, *, slow_send=False):
        disconnected = asyncio.Event()
        delivered = asyncio.Event()
        sent_body = False
        raw = json.dumps(data(stream=True)).encode()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "http_version": "1.1", "scheme": "http", "method": "POST", "root_path": "",
                 "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions", "query_string": b"",
                 "headers": [(b"content-type", b"application/json")], "client": ("127.0.0.1", 1),
                 "server": ("router", 80)}

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                delivered.set()
                if slow_send:
                    await asyncio.Event().wait()

        task = asyncio.create_task(self.app(scope, receive, send))
        await asyncio.wait_for(delivered.wait(), 2)
        return task, disconnected

    async def test_waiting_http_cancellation_removes_request_without_switching(self):
        self.upstream.hold_stream = True
        active, disconnected = await self.asgi_stream()
        waiting = asyncio.create_task(self.client.post(
            "/v1/chat/completions", json=data(HIGH_MODEL),
        ))
        try:
            await until(lambda: self.app.state.manager.snapshot()["queued"] == 1)
            waiting.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiting
            self.assertEqual(self.app.state.manager.snapshot()["queued"], 0)
            self.assertEqual(self.engine.calls, [("start", LOW_MODEL)])
            self.assertEqual(len(self.upstream.requests), 1)
        finally:
            disconnected.set()
            await asyncio.wait_for(active, 2)

    async def test_stream_disconnect_closes_upstream_before_releasing_lease(self):
        self.upstream.hold_stream = True
        task, disconnected = await self.asgi_stream()
        self.assertEqual(self.app.state.manager.snapshot()["active"], 1)
        disconnected.set()
        await asyncio.wait_for(task, 2)
        await asyncio.wait_for(self.upstream.disconnected.wait(), 2)
        await until(lambda: self.engine.model is None)
        self.assertEqual(self.app.state.manager.snapshot()["active"], 0)
        self.assertIn(("stop", LOW_MODEL), self.engine.calls)

    async def test_backpressure_deadline_closes_upstream_and_releases_lease(self):
        self.upstream.hold_stream = True
        with patch.object(router, "SEND_TIMEOUT", 0.02):
            task, _ = await self.asgi_stream(slow_send=True)
            with self.assertRaises((TimeoutError, ExceptionGroup)):
                await asyncio.wait_for(task, 2)
        await asyncio.wait_for(self.upstream.disconnected.wait(), 2)
        self.assertEqual(self.app.state.manager.snapshot()["active"], 0)


if __name__ == "__main__":
    unittest.main()
