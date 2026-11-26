import asyncio
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from starlette.requests import ClientDisconnect

from src.model_router import RoutingRequest, create_app, _sse
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

    def request(self, model=LOW_MODEL):
        return RoutingRequest(model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=1)

    async def test_policy_selection_and_admission_are_atomic(self):
        manager = self.app.state.manager
        small = [manager.workers[key] for key in ("small-0", "small-1")]
        for policy, expected in (
            ("least-inflight", ["small-0", "small-1"] * 4),
            ("round-robin", ["small-0", "small-1"] * 4),
            ("weighted-capacity", ["small-0", "small-0", "small-1", "small-0"] * 2),
        ):
            manager.config = replace(manager.config, policy=policy)
            manager.workers["small-0"].definition = replace(small[0].definition, capacity_weight=3)
            manager.routing_cursor.clear()
            manager.routing_credits.clear()
            selected = []
            for index in range(8):
                lease = manager.submit(self.request(), f"{policy}-{index}")
                selected.append(lease.worker.definition.id)
                self.assertLessEqual(len(lease.worker.active), lease.worker.definition.max_sequences)
                self.assertLessEqual(len(lease.worker.queue), manager.config.queue_capacity_per_worker)
            if policy != "weighted-capacity":
                self.assertEqual(selected, expected)
            else:
                self.assertEqual(selected[:4], expected[:4])
                self.assertEqual(sorted(selected).count("small-0"), 4)
            with self.assertRaises(router.RouterError) as failure:
                manager.submit(self.request(), "overflow")
            self.assertEqual((failure.exception.status, failure.exception.code), (429, "server_busy"))
            for worker in small:
                for lease in list(worker.active) + list(worker.queue):
                    manager.cancel_waiting(lease)
                self.assertEqual(worker.inflight, 0)
            await asyncio.sleep(0)
        self.assertEqual(manager.metrics()["routing"]["policy"], "weighted-capacity")

    async def test_eligibility_queue_depth_and_lifecycle(self):
        manager = self.app.state.manager
        manager.config = replace(manager.config, policy="least-queue-depth")
        leases = [manager.submit(self.request(), str(i)) for i in range(5)]
        self.assertEqual([lease.worker.definition.id for lease in leases],
                         ["small-0", "small-0", "small-0", "small-1", "small-1"])
        manager.workers["small-1"].state = "draining"
        self.assertEqual(manager.submit(self.request(), "next").worker.definition.id, "small-0")
        manager.workers["large-0"].state = "unavailable"
        with self.assertRaises(router.RouterError) as unavailable:
            manager.submit(self.request(HIGH_MODEL), "large")
        self.assertEqual((unavailable.exception.status, unavailable.exception.code), (503, "not_ready"))
        for worker in manager.workers.values():
            for lease in list(worker.active) + list(worker.queue):
                manager.cancel_waiting(lease)
        await asyncio.sleep(0)
        self.assertTrue(all(w.inflight == 0 for w in manager.workers.values()))

    async def test_concurrent_admissions_and_prepare_failure_release(self):
        manager = self.app.state.manager
        manager.config = replace(manager.config, policy="round-robin")

        async def submit(index):
            try:
                return manager.submit(self.request(), str(index))
            except router.RouterError as error:
                return error

        results = await asyncio.gather(*(submit(i) for i in range(20)))
        leases = [result for result in results if not isinstance(result, router.RouterError)]
        self.assertEqual(len(leases), 8)
        self.assertEqual([result.code for result in results if isinstance(result, router.RouterError)],
                         ["server_busy"] * 12)
        self.assertTrue(all(lease.worker.definition.model == LOW_MODEL for lease in leases))
        self.assertEqual(manager.routing_cursor[LOW_MODEL], leases[-1].worker.definition.id)
        for lease in leases:
            manager.cancel_waiting(lease)
        await asyncio.sleep(0)
        self.assertTrue(all(worker.inflight == 0 for worker in manager.workers.values()))

        worker = manager.workers["small-0"]
        async def failed_prepare(*args):
            raise RuntimeError("simulated preparation failure")
        worker.engine.prepare = failed_prepare
        lease = manager.submit(self.request(), "prepare-failure")
        with self.assertRaises(router.RouterError) as failure:
            await lease.ready
        self.assertEqual(failure.exception.code, "engine_failed")
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.inflight, 0)
        self.assertEqual(manager.submit(self.request(), "survivor").worker.definition.id, "small-1")
        for lease in list(manager.workers["small-1"].active):
            manager.cancel_waiting(lease)
        await asyncio.sleep(0)

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
        self.assertEqual((await self.client.get("/ready")).json(), {"status": "ready"})

    async def test_explicit_mode_starts_the_configured_worker_pool(self):
        app = create_app(config=self.config, worker_engine_factory=FakeWorkerEngine, mode="local")
        async with app.router.lifespan_context(app):
            manager = app.state.manager
            self.assertEqual(set(manager.workers), {row.id for row in self.config.workers})
            self.assertTrue(manager.readiness()["ready"])

    async def test_stream_deadline_reports_request_timeout_and_keeps_replica(self):
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
            self.assertEqual(worker.state, "ready")
            self.assertEqual(worker.engine.starts, 1)
            self.assertEqual(worker.engine.stops, 0)
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
                worker.state = "unavailable"
            return result

        with patch.object(router, "_connected", quarantine_after_admission):
            response = await self.post(LOW_MODEL)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "not_ready")
        self.assertEqual(self.forwarded, [])
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.inflight, 0)

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

    async def test_public_readiness_is_minimal_and_requires_every_model(self):
        response = await self.client.get("/ready")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})
        manager = self.app.state.manager

        manager.workers["small-0"].state = "unavailable"
        degraded = await self.client.get("/ready")
        self.assertEqual(degraded.status_code, 200)
        self.assertEqual(degraded.json(), {"status": "ready"})

        manager.workers["large-0"].state = "unavailable"
        missing = await self.client.get("/ready")
        self.assertEqual(missing.status_code, 503)
        self.assertEqual(missing.json(), {"status": "not_ready"})
        serialized = json.dumps(missing.json())
        for private in ("GPU-", "127.0.0.1", "cuda_device", "physical_gpu_uuid",
                        "host", "error", "endpoint", "revision", "queued", HIGH_MODEL):
            self.assertNotIn(private, serialized)

        metrics = (await self.client.get("/metrics", headers={"authorization": "Bearer CLIENT-SECRET"})).json()
        self.assertEqual(metrics["workers"]["small-0"]["state"], "unavailable")
        self.assertEqual(metrics["workers"]["large-0"]["state"], "unavailable")
        self.assertEqual(metrics["models"][LOW_MODEL]["ready"], 1)
        self.assertEqual(metrics["models"][HIGH_MODEL]["ready"], 0)

        manager.workers["large-0"].state = "ready"
        manager.workers["small-1"].state = "unavailable"
        offline = await self.client.get("/ready")
        self.assertEqual(offline.status_code, 503)
        self.assertEqual(offline.json(), {"status": "not_ready"})


    async def test_asgi_send_timing_worker_correlation_and_partial_failure(self):
        async def exchange(*, fail_terminal=False):
            incoming = asyncio.Queue()
            incoming.put_nowait({"type": "http.request", "body": json.dumps({
                "model": LOW_MODEL, "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1, "stream": True,
            }).encode(), "more_body": False})
            waiting, release = asyncio.Event(), asyncio.Event()
            sent = []
            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1", "method": "POST", "scheme": "http",
                "path": "/v1/chat/completions", "raw_path": b"/v1/chat/completions",
                "query_string": b"", "headers": [(b"content-type", b"application/json"),
                                                (b"authorization", b"Bearer CLIENT-SECRET")],
                "client": ("127.0.0.1", 1234), "server": ("gateway", 80),
                "app": self.app,
            }

            async def send(message):
                body = message.get("body", b"")
                if b'"content": "hello"' in body:
                    waiting.set()
                    await release.wait()
                if fail_terminal and body == b"data: [DONE]\n\n":
                    raise OSError("client disappeared")
                sent.append(message)

            with self.assertLogs("inference.router.lifecycle") as captured:
                task = asyncio.create_task(self.app(scope, incoming.get, send))
                try:
                    await asyncio.wait_for(waiting.wait(), 2)
                    metrics = self.app.state.manager.metrics()
                    self.assertEqual(metrics["http"]["first_content_ms"]["count"],
                                     1 if fail_terminal else 0)
                finally:
                    release.set()
                    if fail_terminal:
                        with self.assertRaises(ClientDisconnect):
                            await asyncio.wait_for(task, 3)
                    else:
                        await asyncio.wait_for(task, 3)
            rows = [json.loads(line[line.index("{"):]) for line in captured.output]
            self.assertEqual(len(rows), 1)
            request_id = dict(sent[0]["headers"])[b"x-request-id"].decode()
            self.assertEqual(rows[0]["request_id"], request_id)
            self.assertEqual(rows[0]["worker_id"], "small-0")
            self.assertGreater(rows[0]["first_content_ms"], 0)
            return rows[0]

        delivered = await exchange()
        self.assertEqual(delivered["outcome"], "completed")
        self.assertEqual(delivered["completion_tokens"], 1)
        partial = await exchange(fail_terminal=True)
        self.assertEqual(partial["outcome"], "disconnected")
        metrics = self.app.state.manager.metrics()
        self.assertEqual(metrics["models"][LOW_MODEL]["http"]["completed_tokens"]["completion"], 1)
        self.assertEqual(metrics["workers"]["small-0"]["http"]["requests"]["disconnected"], 1)
        self.assertEqual(metrics["http"]["first_content_ms"]["count"], 2)
        self.assertEqual(metrics["http"]["http_e2e_ms"]["count"], 2)
        self.assertNotIn("internal-small-0", json.dumps(metrics))


if __name__ == "__main__":
    unittest.main()
