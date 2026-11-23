import asyncio
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from src.model_router import create_app, _sse
from src import model_router as router
from src.serving_config import load_config
from src.vllm_engine import HIGH_MODEL, LOW_MODEL

ROOT = Path(__file__).resolve().parents[1]


class FakeWorkerEngine:
    def __init__(self, definition):
        self.definition = definition
        self.base_url = definition.endpoint
        self.model = None
        self.alive = False
        self.starts = 0
        self.stops = 0

    def is_alive(self):
        return self.alive

    async def start(self, model):
        self.starts += 1
        self.model, self.alive = model, True

    async def prepare(self, model, messages, max_tokens):
        return {"prompt_tokens": 3, "prompt_token_ids": [10, 11, 12],
                "eos_token_ids": [2], "formatted_prompt_sha256": "prompt",
                "chat_template_sha256": "template"}

    async def stop(self):
        self.stops += 1
        self.model, self.alive = None, False


class SlowDelivery:
    """An ASGI wrapper that makes marked response frames take real time to deliver."""

    def __init__(self, app, delay, marker=b"x-slow-delivery"):
        self.app, self.delay, self.marker = app, delay, marker

    async def __call__(self, scope, receive, send):
        slow_request = any(key == self.marker for key, _ in scope.get("headers", []))

        async def slow(message):
            if slow_request and message["type"] in {"http.response.start", "http.response.body"}:
                await asyncio.sleep(self.delay)
            await send(message)

        await self.app(scope, receive, slow)


class MultiWorkerGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = load_config(ROOT / "configs/serving-b.json")
        self.tokens = {row.service_token_env: f"internal-{row.id}" for row in self.config.workers}
        self.tokens["INFERENCE_API_TOKEN"] = "CLIENT-SECRET"
        self.saved_env = patch.dict(os.environ, self.tokens)
        self.saved_env.start()
        self.addCleanup(self.saved_env.stop)
        self.forwarded = []

        async def upstream(request):
            body = json.loads(request.content)
            self.forwarded.append((request.url.host, request.headers.get("authorization"), body["model"]))
            common = {"id": "chatcmpl-" + request.headers["x-request-id"],
                      "object": "chat.completion.chunk", "created": 1, "model": body["model"]}
            usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
            if body["stream"]:
                frames = [
                    {**common, "prompt_token_ids": [10, 11, 12], "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None, "token_ids": [13]}]},
                    {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                    {**common, "choices": [], "usage": usage},
                ]
                return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                      content="".join(_sse(frame) for frame in frames) + "data: [DONE]\n\n")
            return httpx.Response(200, json={**common, "object": "chat.completion",
                "prompt_token_ids": [10, 11, 12], "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "hello"}, "finish_reason": "length",
                "token_ids": [13]}], "usage": usage})

        self.app = create_app(config=self.config, worker_engine_factory=FakeWorkerEngine)
        self.lifespan = self.app.router.lifespan_context(self.app)
        await self.lifespan.__aenter__()
        await self.app.state.client.aclose()
        self.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False)
        self.addAsyncCleanup(self.app.state.client.aclose)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://gateway")
        self.addAsyncCleanup(self.client.aclose)
        self.addAsyncCleanup(self.lifespan.__aexit__, None, None, None)

    async def post(self, model, *, stream=False, token="CLIENT-SECRET"):
        return await self.client.post("/v1/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1, "stream": stream,
        }, headers={"authorization": f"Bearer {token}"})

    async def test_model_dispatch_credentials_and_openai_response(self):
        for model in (LOW_MODEL, HIGH_MODEL):
            response = await self.post(model)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["model"], model)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "hello")
            self.assertEqual(response.json()["usage"]["total_tokens"], 4)
        self.assertEqual([row[2] for row in self.forwarded], [LOW_MODEL, HIGH_MODEL])
        self.assertEqual([row[1] for row in self.forwarded], ["Bearer internal-small-0", "Bearer internal-large-0"])
        self.assertNotIn("CLIENT-SECRET", str(self.forwarded))
        self.assertTrue((await self.client.get("/ready")).json()["ready"])

    async def test_explicit_mode_starts_the_configured_worker_pool(self):
        app = create_app(config=self.config, worker_engine_factory=FakeWorkerEngine, mode="local")
        async with app.router.lifespan_context(app):
            manager = app.state.manager
            self.assertEqual(set(manager.workers), {row.id for row in self.config.workers})
            self.assertTrue(manager.readiness()["models"][LOW_MODEL]["ready"])
            self.assertTrue(manager.readiness()["models"][HIGH_MODEL]["ready"])

    async def test_stream_deadline_reports_request_timeout_and_recycles_replica(self):
        config = replace(self.config, workers=tuple(
            replace(row, request_timeout_s=0.1) for row in self.config.workers))
        stall = asyncio.Event()

        async def stalling_upstream(request):
            body = json.loads(request.content)
            self.forwarded.append((request.url.host, request.headers.get("authorization"), body["model"]))
            common = {"id": "chatcmpl-" + request.headers["x-request-id"],
                      "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                      "prompt_token_ids": [10, 11, 12]}
            first = {**common, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"},
                                            "finish_reason": None, "token_ids": [13]}]}

            async def body_stream():
                yield _sse(first).encode()
                await stall.wait()

            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body_stream())

        app = create_app(config=config, worker_engine_factory=FakeWorkerEngine)
        async with app.router.lifespan_context(app):
            await app.state.client.aclose()
            app.state.client = httpx.AsyncClient(
                transport=httpx.MockTransport(stalling_upstream), trust_env=False)
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gateway")
            try:
                response = await client.post("/v1/chat/completions", json={
                    "model": LOW_MODEL, "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 1, "stream": True,
                }, headers={"authorization": "Bearer CLIENT-SECRET"})
            finally:
                stall.set()
                await client.aclose()
            await app.state.client.aclose()
            self.assertEqual(response.status_code, 200)
            self.assertIn('"code": "request_timeout"', response.text)
            manager = app.state.manager
            worker = manager.workers["small-0"]
            async with asyncio.timeout(2):
                while worker.state != "ready":
                    await asyncio.sleep(0.01)
            self.assertEqual(worker.engine.starts, 2)
            self.assertEqual(worker.engine.stops, 1)
            self.assertIsNone(worker.last_error)

    async def test_quarantine_between_admission_and_dispatch_returns_not_ready(self):
        manager = self.app.state.manager
        worker = manager.workers["small-0"]
        connected = router._connected
        calls = 0

        async def quarantine_after_admission(request, future):
            nonlocal calls
            result = await connected(request, future)
            if calls == 0:
                calls = 1
                worker.state = "suspect"
            return result

        with patch.object(router, "_connected", quarantine_after_admission):
            response = await self.post(LOW_MODEL)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "not_ready")
        self.assertEqual(self.forwarded, [])
        self.assertEqual(worker.state, "suspect")
        self.assertEqual(worker.inflight, 0)

    async def test_slow_sibling_delivery_keeps_a_recycled_worker_routable(self):
        config = load_config(ROOT / "configs/serving-a.json")
        config = replace(config, health_interval_s=60, workers=tuple(
            replace(row, request_timeout_s=0.05) for row in config.workers))
        dispatched = []

        async def upstream(request):
            body = json.loads(request.content)
            dispatched.append(body["stream"])
            if not body["stream"]:
                return httpx.Response(500)
            common = {"id": "chatcmpl-" + request.headers["x-request-id"],
                      "object": "chat.completion.chunk", "created": 1, "model": body["model"],
                      "prompt_token_ids": [10, 11, 12]}
            frames = [
                {**common, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"},
                                        "finish_reason": None, "token_ids": [13]}]},
                {**common, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]},
                {**common, "choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}},
            ]

            async def stalled():
                await asyncio.sleep(1.0)
                for frame in frames:
                    yield _sse(frame).encode()
                yield b"data: [DONE]\n\n"

            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stalled())

        body = {"model": LOW_MODEL, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1}
        auth = {"authorization": "Bearer CLIENT-SECRET"}
        app = create_app(config=config, worker_engine_factory=FakeWorkerEngine, mode="local")
        async with app.router.lifespan_context(app):
            await app.state.client.aclose()
            app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False)
            client = httpx.AsyncClient(transport=httpx.ASGITransport(app=SlowDelivery(app, 0.15)),
                                       base_url="http://gateway", timeout=30)
            slow_auth = {**auth, "x-slow-delivery": "1"}
            try:
                with patch("src.model_router.SEND_TIMEOUT", 0.2), \
                        patch("src.model_router.CLOSE_TIMEOUT", 0.1), \
                        patch("src.worker_manager.CLOSE_TIMEOUT", 0.1), \
                        patch("src.worker_manager.DRAIN_TIMEOUT", 0.01):
                    streamed = asyncio.create_task(client.post(
                        "/v1/chat/completions", json={**body, "stream": True}, headers=slow_auth))
                    async with asyncio.timeout(5):
                        while not dispatched:
                            await asyncio.sleep(0.01)
                    failed = await client.post("/v1/chat/completions", json=body, headers=auth)
                    response = await streamed
            finally:
                await client.aclose()
            await app.state.client.aclose()
            manager = app.state.manager
            worker = manager.workers["small-0"]
            async with asyncio.timeout(10):
                while worker.cleanup_task is not None or worker.state != "ready":
                    await asyncio.sleep(0.01)
            self.assertEqual(worker.state, "ready")
            self.assertIsNone(worker.last_error)
            self.assertEqual(worker.requests["timeout"], 1)
            self.assertEqual(worker.requests["failed"], 1)
            self.assertEqual(worker.engine.starts, 2)
            self.assertEqual(worker.engine.stops, 1)
        self.assertEqual(failed.status_code, 502)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"code": "request_timeout"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    async def test_gateway_auth_rejects_before_worker_dispatch(self):
        response = await self.post(LOW_MODEL, token="wrong-token")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["code"], "unauthorized")
        self.assertEqual(self.forwarded, [])

    async def test_streaming_terminal_usage_and_worker_metrics(self):
        response = await self.post(LOW_MODEL, stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"content": "hello"', response.text)
        self.assertIn('"total_tokens": 4', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))
        metrics = (await self.client.get("/metrics", headers={"authorization": "Bearer CLIENT-SECRET"})).json()
        self.assertEqual(metrics["models"][LOW_MODEL]["requests"]["completed"], 1)
        self.assertEqual(len(metrics["physical_gpus"]), 2)
        self.assertTrue(all(row["utilization_percent"] is None for row in metrics["physical_gpus"]))
        self.assertNotIn("internal-small-0", json.dumps(metrics))

    async def test_requested_model_with_no_ready_worker_returns_sanitized_503(self):
        manager = self.app.state.manager
        for worker in manager.workers.values():
            if worker.definition.model == HIGH_MODEL:
                worker.state = "unavailable"
        response = await self.post(HIGH_MODEL)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "not_ready")
        self.assertEqual(len(self.forwarded), 0)

    async def test_public_readiness_hides_worker_topology_and_reports_routable_models(self):
        response = await self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), {"ready", "models"})
        self.assertTrue(body["ready"])
        serialized = json.dumps(body)
        for private in ("GPU-", "127.0.0.1", "cuda_device", "physical_gpu_uuid",
                        "host", "error", "endpoint", "revision", "queued"):
            self.assertNotIn(private, serialized)
        manager = self.app.state.manager
        manager.workers["large-0"].state = "unavailable"
        partial = await self.client.get("/ready")
        self.assertEqual(partial.status_code, 200)
        self.assertTrue(partial.json()["ready"])
        self.assertEqual(partial.json()["models"][HIGH_MODEL]["status"], "unavailable")
        self.assertEqual(partial.json()["models"][LOW_MODEL]["status"], "ready")
        manager.workers["large-0"].state = "ready"
        manager.workers["small-0"].state = "unavailable"
        degraded = await self.client.get("/ready")
        self.assertEqual(degraded.status_code, 200)
        self.assertTrue(degraded.json()["ready"])
        self.assertEqual(degraded.json()["models"][LOW_MODEL]["status"], "degraded")
        self.assertEqual(degraded.json()["models"][LOW_MODEL]["ready"], 1)
        self.assertEqual(degraded.json()["models"][HIGH_MODEL]["status"], "ready")
        metrics = (await self.client.get("/metrics", headers={"authorization": "Bearer CLIENT-SECRET"})).json()
        self.assertEqual(metrics["workers"]["small-0"]["state"], "unavailable")
        for worker in manager.workers.values():
            worker.state = "unavailable"
        offline = await self.client.get("/ready")
        self.assertEqual(offline.status_code, 503)
        self.assertFalse(offline.json()["ready"])
        self.assertEqual(offline.json()["models"][LOW_MODEL]["status"], "unavailable")
        self.assertEqual(offline.json()["models"][HIGH_MODEL]["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
