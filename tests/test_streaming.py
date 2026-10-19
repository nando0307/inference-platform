import asyncio
import json
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx

from src.api import app
from src.inference import ContextLimitError, GenerationResult, MODEL_ID


def payload(stream=True):
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


class Connection:
    """An ASGI connection that exposes frames immediately, without HTTPX buffering."""

    def __init__(self, stream=True, spec="2.4"):
        self.incoming = asyncio.Queue()
        self.incoming.put_nowait({
            "type": "http.request", "body": json.dumps(payload(stream)).encode(),
            "more_body": False,
        })
        self.messages = []
        self.headers_sent = asyncio.Event()
        self.content_sent = asyncio.Event()
        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": spec},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
            "query_string": b"", "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234), "server": ("test", 80),
        }
        self.task = asyncio.create_task(app(scope, self.incoming.get, self.send))

    async def send(self, message):
        self.messages.append(message)
        if message["type"] == "http.response.start":
            self.headers_sent.set()
        elif message.get("body", b"").startswith(b"data: {"):
            data = json.loads(message["body"][6:])
            if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                self.content_sent.set()

    def disconnect(self):
        self.incoming.put_nowait({"type": "http.disconnect"})

    def body(self):
        return b"".join(message.get("body", b"") for message in self.messages)

    def events(self):
        return [
            "[DONE]" if frame == b"data: [DONE]" else json.loads(frame[6:])
            for frame in self.body().split(b"\n\n") if frame
        ]


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        loader_patch = patch(
            "src.api.load_runtime", return_value=SimpleNamespace(tokenizer=Mock())
        )
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        self.addAsyncCleanup(context.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.result = GenerationResult("你好, world!", 12, [7, 8, 2], "stop")

    async def test_incremental_content_and_truthful_terminal_usage(self):
        release = Event()

        def held_generation(*args, streamer, on_prepared, **kwargs):
            on_prepared()
            streamer.on_finalized_text("")
            streamer.on_finalized_text("你好, ")
            if not release.wait(5):
                raise TimeoutError("Producer was not released")
            streamer.on_finalized_text("world!", stream_end=True)
            return self.result

        with patch("src.api.generate", side_effect=held_generation):
            connection = Connection()
            try:
                await asyncio.wait_for(connection.content_sent.wait(), 2)
                self.assertFalse(connection.task.done())
                self.assertTrue(app.state.admission_guard.locked())
                self.assertNotIn("[DONE]", connection.events())
                busy = await self.client.post("/v1/chat/completions", json=payload())
                self.assertEqual(busy.status_code, 429)
            finally:
                release.set()
                await asyncio.wait_for(connection.task, 3)
        events = connection.events()
        self.assertEqual(events[-1], "[DONE]")
        self.assertEqual(events.count("[DONE]"), 1)
        chunks = events[:-1]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(
            "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks),
            self.result.text,
        )
        terminal = chunks[-1]
        self.assertEqual(terminal["choices"][0]["delta"], {})
        self.assertEqual(terminal["choices"][0]["finish_reason"], "stop")
        self.assertEqual(terminal["usage"], {
            "prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15,
        })
        for chunk in chunks[:-1]:
            self.assertIsNone(chunk["choices"][0]["finish_reason"])
            self.assertNotIn("usage", chunk)
            self.assertNotEqual(chunk["choices"][0]["delta"].get("content"), "")
        for field in ("id", "model", "created", "object"):
            self.assertEqual(len({chunk[field] for chunk in chunks}), 1)
        headers = dict(connection.messages[0]["headers"])
        self.assertEqual(connection.messages[0]["status"], 200)
        self.assertTrue(headers[b"content-type"].startswith(b"text/event-stream"))
        self.assertEqual(headers[b"cache-control"], b"no-cache")
        self.assertEqual(chunks[0]["id"], "chatcmpl-" + headers[b"x-request-id"].decode())

    async def test_queue_timeout_is_not_a_generation_failure(self):
        release = Event()

        def delayed_generation(*args, streamer, on_prepared, **kwargs):
            on_prepared()
            if not release.wait(5):
                raise TimeoutError("Producer was not released")
            streamer.on_finalized_text(self.result.text, stream_end=True)
            return self.result

        with patch("src.api.generate", side_effect=delayed_generation):
            connection = Connection()
            try:
                await asyncio.wait_for(connection.headers_sent.wait(), 2)
                await asyncio.sleep(0.25)  # Cross multiple 0.1-second queue timeouts.
                self.assertFalse(connection.task.done())
                self.assertNotIn(b'"error"', connection.body())
                self.assertEqual((await self.client.get("/health")).status_code, 200)
            finally:
                release.set()
                await asyncio.wait_for(connection.task, 3)
        self.assertEqual(connection.events()[-1], "[DONE]")

    async def test_preparation_failure_is_json_before_headers_and_recovers(self):
        for failure, status in (
            (ContextLimitError("Prompt tokens (2040) plus output (100) exceed 2048."), 400),
            (RuntimeError("PRIVATE-PREPARATION"), 500),
        ):
            with self.subTest(status=status):
                with patch("src.api.generate", side_effect=failure):
                    response = await asyncio.wait_for(
                        self.client.post("/v1/chat/completions", json=payload()), 2
                    )
                self.assertEqual(response.status_code, status)
                self.assertTrue(response.headers["content-type"].startswith("application/json"))
                self.assertNotIn("PRIVATE-PREPARATION", response.text)
                self.assertFalse(app.state.admission_guard.locked())
                with patch("src.api.generate", return_value=self.result):
                    recovered = await self.client.post("/v1/chat/completions", json=payload(False))
                self.assertEqual(recovered.status_code, 200)

    async def test_stream_submission_failure_does_not_leave_a_preparation_waiter(self):
        with patch.object(app.state.executor, "submit", side_effect=RuntimeError("PRIVATE")):
            response = await asyncio.wait_for(
                self.client.post("/v1/chat/completions", json=payload()), 2
            )
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("PRIVATE", response.text)
        self.assertFalse(app.state.admission_guard.locked())

    async def test_failure_with_or_without_sentinel_has_no_success_terminal(self):
        for sentinel in (False, True):
            with self.subTest(sentinel=sentinel):
                release = Event()

                def failing_generation(*args, streamer, on_prepared, **kwargs):
                    on_prepared()
                    streamer.on_finalized_text("Partial text ", stream_end=sentinel)
                    if not release.wait(5):
                        raise TimeoutError("Producer was not released")
                    raise RuntimeError("PRIVATE-GENERATION")

                with patch("src.api.generate", side_effect=failing_generation):
                    connection = Connection()
                    try:
                        await asyncio.wait_for(connection.content_sent.wait(), 2)
                        self.assertFalse(connection.task.done())
                        self.assertNotIn("[DONE]", connection.events())
                    finally:
                        release.set()
                        await asyncio.wait_for(connection.task, 3)
                events = connection.events()
                self.assertEqual(events[-1], "[DONE]")
                self.assertEqual(events.count("[DONE]"), 1)
                error = events[-2]["error"]
                self.assertEqual(error["code"], "generation_failed")
                self.assertNotIn(b"PRIVATE-GENERATION", connection.body())
                self.assertEqual(connection.messages[0]["status"], 200)
                self.assertEqual(error["request_id"], dict(connection.messages[0]["headers"])[b"x-request-id"].decode())
                for event in events[:-1]:
                    self.assertNotIn("usage", event)
                    if "choices" in event:
                        self.assertIsNone(event["choices"][0]["finish_reason"])
                with patch("src.api.generate", return_value=self.result):
                    recovered = await self.client.post("/v1/chat/completions", json=payload(False))
                self.assertEqual(recovered.status_code, 200)

    async def test_disconnect_before_preparation_or_ordinary_result_sends_nothing(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                started = Event()
                release = Event()

                def held_generation(*args, **kwargs):
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("Producer was not released")
                    raise RuntimeError("Failure after disconnect")

                with patch("src.api.generate", side_effect=held_generation):
                    connection = Connection(stream)
                    producer = None
                    try:
                        self.assertTrue(await asyncio.to_thread(started.wait, 2))
                        producer = next(iter(app.state.producer_futures))
                        connection.disconnect()
                        await asyncio.wait_for(connection.task, 2)
                        self.assertEqual(connection.messages, [])
                        self.assertFalse(producer.done())
                        self.assertTrue(app.state.admission_guard.locked())
                        busy = await self.client.post("/v1/chat/completions", json=payload(False))
                        self.assertEqual(busy.status_code, 429)
                    finally:
                        release.set()
                        await asyncio.wait_for(connection.task, 3)
                        if producer is not None:
                            await asyncio.gather(producer, return_exceptions=True)
                self.assertFalse(app.state.admission_guard.locked())

    async def test_disconnect_after_content_keeps_ownership_for_both_asgi_versions(self):
        for spec in ("2.3", "2.4"):
            with self.subTest(spec=spec):
                release = Event()

                def held_generation(*args, streamer, on_prepared, **kwargs):
                    on_prepared()
                    streamer.on_finalized_text("Partial text ")
                    if not release.wait(5):
                        raise TimeoutError("Producer was not released")
                    streamer.on_finalized_text("remaining", stream_end=True)
                    return self.result

                with patch("src.api.generate", side_effect=held_generation):
                    connection = Connection(spec=spec)
                    producer = None
                    try:
                        await asyncio.wait_for(connection.content_sent.wait(), 2)
                        producer = next(iter(app.state.producer_futures))
                        connection.disconnect()
                        await asyncio.wait_for(connection.task, 2)
                        self.assertNotIn("[DONE]", connection.events())
                        self.assertFalse(producer.done())
                        self.assertTrue(app.state.admission_guard.locked())
                        self.assertEqual((await self.client.get("/ready")).status_code, 503)
                        busy = await self.client.post("/v1/chat/completions", json=payload())
                        self.assertEqual(busy.status_code, 429)
                    finally:
                        release.set()
                        await asyncio.wait_for(connection.task, 3)
                        if producer is not None:
                            await asyncio.wait_for(asyncio.shield(producer), 3)
                self.assertEqual((await self.client.get("/ready")).status_code, 200)


if __name__ == "__main__":
    unittest.main()
