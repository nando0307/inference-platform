import asyncio
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from src.model_router import RouterError, RoutingRequest, _complete
from src.serving_config import load_config
from src.worker_manager import MultiWorkerManager
from src.vllm_engine import HIGH_MODEL, LOW_MODEL

ROOT = Path(__file__).resolve().parents[1]


def request(model=LOW_MODEL):
    return RoutingRequest.model_validate({
        "model": model, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1,
    })


class FakeEngine:
    def __init__(self, definition, *, gate=None, fail=False):
        self.definition, self.base_url = definition, definition.endpoint
        self.model = None
        self.alive = False
        self.gate = gate
        self.fail = fail
        self.starts = 0
        self.stops = 0

    def is_alive(self):
        return self.alive

    async def start(self, model):
        self.starts += 1
        if self.fail:
            raise RuntimeError("private startup detail")
        self.model, self.alive = model, True

    async def prepare(self, model, messages, max_tokens):
        if self.gate:
            await self.gate.wait()
        return {"prompt_tokens": 1, "prompt_token_ids": [1], "eos_token_ids": [2]}

    async def stop(self):
        self.stops += 1
        self.model, self.alive = None, False


class WorkerManagerTests(unittest.IsolatedAsyncioTestCase):
    def config(self, placement):
        return load_config(ROOT / f"configs/serving-{placement.lower()}.json")

    async def test_partial_startup_serves_only_ready_exact_model(self):
        engines = {}
        def factory(definition):
            engine = FakeEngine(definition, fail=definition.model == HIGH_MODEL)
            engines[definition.id] = engine
            return engine

        manager = MultiWorkerManager(self.config("B"), engine_factory=factory)
        self.addAsyncCleanup(manager.shutdown)
        self.assertTrue(await manager.start())
        state = manager.snapshot()
        self.assertFalse(state["ready"])
        self.assertFalse(manager.readiness()["ready"])
        self.assertEqual(state["models"][LOW_MODEL]["status"], "ready")
        self.assertEqual(state["models"][HIGH_MODEL]["status"], "unavailable")
        lease = manager.submit(request(LOW_MODEL), "small-request")
        await lease.ready
        self.assertEqual(lease.worker.definition.id, "small-0")
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(HIGH_MODEL), "large-request")
        self.assertEqual(caught.exception.status, 503)
        self.assertNotIn("private", str(caught.exception).lower())
        await manager.release(lease, True)

    async def test_least_inflight_replica_routing_is_model_exact_and_stable(self):
        manager = MultiWorkerManager(self.config("B"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        first = manager.submit(request(LOW_MODEL), "first")
        second = manager.submit(request(LOW_MODEL), "second")
        large = manager.submit(request(HIGH_MODEL), "large")
        await asyncio.gather(first.ready, second.ready, large.ready)
        self.assertEqual(first.worker.definition.id, "small-0")
        self.assertEqual(second.worker.definition.id, "small-1")
        self.assertEqual(large.worker.definition.model, HIGH_MODEL)
        self.assertNotEqual(large.worker.definition.id, first.worker.definition.id)
        await asyncio.gather(manager.release(first, True), manager.release(second, True), manager.release(large, True))

    async def test_two_waiting_slots_backpressure_and_queued_cancellation(self):
        gate = asyncio.Event()
        manager = MultiWorkerManager(self.config("D"), engine_factory=lambda row: FakeEngine(row, gate=gate))
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        leases = [manager.submit(request(), f"request-{n}") for n in range(4)]
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "overflow")
        self.assertEqual(caught.exception.status, 429)
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        self.assertEqual(len(worker.active), 2)
        self.assertEqual(len(worker.queue), 2)
        manager.cancel_waiting(leases[-1])
        self.assertEqual(len(worker.queue), 1)
        gate.set()
        await asyncio.gather(leases[0].ready, leases[1].ready)
        await asyncio.gather(manager.release(leases[0], True), manager.release(leases[1], True))
        await leases[2].ready
        await manager.release(leases[2], True)
        self.assertEqual(len(worker.active), 0)

    async def test_saturated_model_counts_one_rejection_per_request(self):
        manager = MultiWorkerManager(self.config("B"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        leases = [manager.submit(request(), f"request-{n}") for n in range(8)]
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "overflow")
        self.assertEqual(caught.exception.status, 429)
        metrics = manager.metrics()
        self.assertEqual(metrics["models"][LOW_MODEL]["requests"]["rejected"], 1)
        self.assertEqual(sum(row["rejected"] for row in metrics["worker_requests"].values()), 1)
        for lease in leases:
            await manager.release(lease, True)

    async def test_cancelled_dispatch_keeps_worker_ready_and_never_restarts_it(self):
        engines = {}
        def factory(row):
            engines[row.id] = FakeEngine(row)
            return engines[row.id]
        manager = MultiWorkerManager(self.config("D"), engine_factory=factory)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        first = manager.submit(request(), "first")
        second = manager.submit(request(), "second")
        queued = manager.submit(request(), "queued")
        await asyncio.gather(first.ready, second.ready)
        self.assertIn(queued, worker.queue)
        first.dispatched = True
        await manager.release(first, False)
        self.assertEqual(worker.state, "ready")
        await queued.ready
        second.dispatched = True
        await manager.release(second, True)
        await manager.release(queued, True)
        self.assertEqual(worker.engine.starts, 1)
        self.assertEqual(worker.engine.stops, 0)
        served = manager.submit(request(), "after-cancel")
        await served.ready
        self.assertIs(served.worker, worker)
        await manager.release(served, True)

    async def test_worker_failure_remains_unavailable_without_automatic_restart(self):
        engines = {}
        def factory(row):
            engines[row.id] = FakeEngine(row)
            return engines[row.id]
        manager = MultiWorkerManager(self.config("B"), engine_factory=factory)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        failed = manager.workers["small-0"]
        lease = manager.submit(request(), "dispatched")
        await lease.ready
        self.assertEqual(lease.worker.definition.id, "small-0")
        lease.dispatched = True
        manager.fail("upstream_failed", exclude=lease)
        await manager.release(lease, False)
        self.assertEqual(failed.state, "unavailable")
        self.assertEqual(failed.engine.starts, 1)
        self.assertEqual(failed.engine.stops, 0)
        replacement = manager.submit(request(), "failover")
        await replacement.ready
        self.assertEqual(replacement.worker.definition.id, "small-1")
        self.assertEqual(manager.workers["large-0"].engine.starts, 1)
        await manager.release(replacement, True)

    async def test_dead_local_worker_leaves_routing_and_same_model_replica_serves(self):
        manager = MultiWorkerManager(self.config("B"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        dead = manager.workers["small-0"]
        dead.engine.alive = False
        await manager._probe(dead)
        self.assertEqual(dead.state, "unavailable")
        self.assertEqual(manager.metrics()["workers"]["small-0"]["error"], "health_check_failed")
        lease = manager.submit(request(), "failover")
        await lease.ready
        self.assertEqual(lease.worker.definition.id, "small-1")
        await manager.release(lease, True)
        other = manager.workers["small-1"]
        other.engine.alive = False
        await manager._probe(other)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "no-replica")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "not_ready")
        state = manager.snapshot()
        self.assertFalse(state["ready"])
        self.assertEqual(state["models"][LOW_MODEL]["status"], "unavailable")
        self.assertEqual(state["models"][HIGH_MODEL]["status"], "ready")
        large = manager.workers["large-0"]
        large.engine.alive = False
        await manager._probe(large)
        self.assertFalse(manager.readiness()["ready"])

    async def test_release_after_local_process_death_never_restarts_worker(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        lease = manager.submit(request(), "dispatched")
        await lease.ready
        worker.engine.alive = False
        lease.dispatched = True
        await manager.release(lease, False)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.engine.starts, 1)
        self.assertEqual(worker.engine.stops, 0)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "after-death")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(manager.snapshot()["models"][LOW_MODEL]["status"], "unavailable")

    async def test_dispatch_requires_current_admission_on_ready_worker(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        fake_request = SimpleNamespace(state=SimpleNamespace(), app=SimpleNamespace(
            state=SimpleNamespace(client=None, manager=manager)))

        async def admitted(name):
            lease = manager.submit(request(), name)
            await lease.ready
            self.assertIs(lease.worker, worker)
            return lease

        stale_state = await admitted("stale-state")
        worker.state = "unavailable"
        with self.assertRaises(RouterError) as caught:
            await _complete(fake_request, stale_state)
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "not_ready")
        self.assertFalse(stale_state.dispatched)
        worker.state = "ready"
        manager.cancel_waiting(stale_state)

        stale_generation = await admitted("stale-generation")
        worker.generation += 1
        with self.assertRaises(RouterError):
            await _complete(fake_request, stale_generation)
        self.assertFalse(stale_generation.dispatched)
        worker.generation -= 1
        manager.cancel_waiting(stale_generation)

        stale_membership = await admitted("stale-membership")
        worker.active.discard(stale_membership)
        with self.assertRaises(RouterError):
            await _complete(fake_request, stale_membership)
        self.assertFalse(stale_membership.dispatched)
        await manager.release(stale_membership, False)

    async def test_remote_cancellation_keeps_worker_ready_without_recycle(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine, mode="remote")
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        lease = manager.submit(request(), "remote-cancel")
        await lease.ready
        lease.dispatched = True
        await manager.release(lease, False)
        self.assertEqual(worker.state, "ready")
        self.assertEqual(worker.engine.starts, 1)
        self.assertEqual(worker.engine.stops, 0)
        again = manager.submit(request(), "after-remote-cancel")
        await again.ready
        await manager.release(again, True)

    async def test_death_during_cancellation_is_not_restarted(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        lease = manager.submit(request(), "canceled")
        await lease.ready
        lease.dispatched = True
        await manager.release(lease, False)
        self.assertEqual(worker.state, "ready")
        worker.engine.alive = False
        failed = manager.submit(request(), "dead")
        await failed.ready
        failed.dispatched = True
        await manager.release(failed, False)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.last_error, "process_exited")
        self.assertEqual(worker.engine.starts, 1)
        self.assertEqual(worker.engine.stops, 0)

    async def test_terminal_outcomes_are_counted_as_observed(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        for outcome in ("completed", "failed", "canceled", "timeout"):
            lease = manager.submit(request(), outcome)
            await lease.ready
            lease.dispatched = True
            await manager.release(lease, outcome == "completed", outcome)
            await asyncio.gather(*manager.tasks)
            self.assertEqual(worker.state, "ready")
        self.assertEqual({key: worker.requests[key]
                          for key in ("completed", "failed", "canceled", "timeout")},
                         {"completed": 1, "failed": 1, "canceled": 1, "timeout": 1})
        model_requests = manager.metrics()["models"][LOW_MODEL]["requests"]
        self.assertEqual(model_requests["timeout"], 1)
        self.assertEqual(model_requests["canceled"], 1)

    async def test_health_failure_counts_each_queued_request_once(self):
        gate = asyncio.Event()
        manager = MultiWorkerManager(
            self.config("D"), engine_factory=lambda row: FakeEngine(row, gate=gate))
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        for n in range(3):
            manager.submit(request(), f"queued-{n}")
        await asyncio.sleep(0)
        self.assertEqual(len(worker.queue), 1)
        worker.engine.alive = False
        await manager._probe(worker, worker.generation)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.requests["failed"], 1)

    async def test_startup_crash_cleans_partial_engine_once_and_preserves_readiness(self):
        class CrashingEngine(FakeEngine):
            async def start(self, model):
                await super().start(model)
                if self.definition.model == LOW_MODEL:
                    raise RuntimeError("private partial startup")

        manager = MultiWorkerManager(self.config("D"), engine_factory=CrashingEngine)
        self.assertTrue(await manager.start())
        failed = manager.workers["small-0"]
        self.assertEqual(failed.state, "unavailable")
        self.assertEqual(failed.last_error, "RuntimeError")
        self.assertFalse(manager.readiness()["ready"])
        self.assertEqual(failed.engine.stops, 1)
        self.assertFalse(failed.engine.is_alive())
        await asyncio.gather(manager.shutdown(), manager.shutdown())
        await manager.shutdown()
        self.assertEqual(failed.engine.stops, 1)
        self.assertEqual(failed.state, "stopped")
        self.assertEqual(manager.workers["large-0"].engine.stops, 1)
        self.assertFalse(manager.tasks)

    async def test_shutdown_during_startup_cancels_load_and_cleans_once(self):
        entered = asyncio.Event()
        class SlowEngine(FakeEngine):
            async def start(self, model):
                self.starts += 1
                entered.set()
                await asyncio.Event().wait()

        manager = MultiWorkerManager(self.config("D"), engine_factory=SlowEngine)
        starting = asyncio.create_task(manager.start())
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(manager.shutdown(), 1)
        with self.assertRaises(asyncio.CancelledError):
            await starting
        self.assertFalse(manager.readiness()["ready"])
        self.assertTrue(all(w.engine.stops == 1 and w.state == "stopped"
                            for w in manager.workers.values()))
        await manager.shutdown()
        self.assertTrue(all(w.engine.stops == 1 for w in manager.workers.values()))

    async def test_health_loss_fails_queue_and_cancels_active_before_shutdown(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        await manager.start()
        worker = manager.workers["small-0"]
        first, second, queued = [manager.submit(request(), str(i)) for i in range(3)]
        await asyncio.gather(first.ready, second.ready)
        async def hold(lease):
            try:
                await asyncio.Event().wait()
            finally:
                await manager.release(lease, False, "canceled")
        first.task, second.task = asyncio.create_task(hold(first)), asyncio.create_task(hold(second))
        await asyncio.sleep(0)
        worker.engine.alive = False
        await manager._probe(worker, worker.generation)
        with self.assertRaises(RouterError):
            await queued.ready
        await asyncio.gather(first.task, second.task, return_exceptions=True)
        await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.requests["failed"], 1)
        self.assertEqual(worker.requests["canceled"], 2)
        self.assertEqual((len(worker.active), len(worker.queue)), (0, 0))
        await manager.shutdown()
        self.assertEqual(worker.engine.stops, 1)
        self.assertFalse(manager.readiness()["ready"])

    async def test_shutdown_drains_cancelled_request_and_fails_queued_once(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        await manager.start()
        worker = manager.workers["small-0"]
        first, second, queued = [manager.submit(request(), str(i)) for i in range(3)]
        await asyncio.gather(first.ready, second.ready)
        draining = asyncio.Event()
        finish = asyncio.Event()
        async def hold(lease):
            try:
                await asyncio.Event().wait()
            finally:
                draining.set()
                await finish.wait()
                await manager.release(lease, False, "canceled")
        first.task, second.task = asyncio.create_task(hold(first)), asyncio.create_task(hold(second))
        await asyncio.sleep(0)
        shutting = asyncio.create_task(manager.shutdown())
        await asyncio.wait_for(draining.wait(), 1)
        self.assertEqual(worker.engine.stops, 0)
        with self.assertRaises(RouterError):
            await queued.ready
        finish.set()
        await asyncio.wait_for(shutting, 1)
        await asyncio.gather(first.task, second.task, return_exceptions=True)
        self.assertEqual(worker.requests["failed"], 1)
        self.assertEqual(worker.requests["canceled"], 2)
        self.assertEqual((len(worker.active), len(worker.queue)), (0, 0))
        self.assertEqual(worker.engine.stops, 1)

    async def test_shutdown_timeout_reports_failure_and_stops_once(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        await manager.start()
        lease = manager.submit(request(), "stuck")
        await lease.ready
        with patch("src.worker_manager.DRAIN_TIMEOUT", 0.01):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                await asyncio.wait_for(manager.shutdown(), 1)
        self.assertFalse(manager.readiness()["ready"])
        self.assertEqual(lease.worker.engine.stops, 1)
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            await manager.shutdown()
        self.assertEqual(lease.worker.engine.stops, 1)
        await manager.release(lease, False)

    async def test_stuck_preparation_cannot_hold_shutdown_forever(self):
        preparing = asyncio.Event()
        finish = asyncio.Event()
        class SlowPrepare(FakeEngine):
            async def prepare(self, model, messages, max_tokens):
                preparing.set()
                try:
                    await finish.wait()
                except asyncio.CancelledError:
                    await finish.wait()
                return await super().prepare(model, messages, max_tokens)

        manager = MultiWorkerManager(self.config("D"), engine_factory=SlowPrepare)
        await manager.start()
        lease = manager.submit(request(), "stuck-prepare")
        await asyncio.wait_for(preparing.wait(), 1)
        with patch("src.worker_manager.DRAIN_TIMEOUT", 0.01):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                await asyncio.wait_for(manager.shutdown(), 1)
        self.assertFalse(manager.readiness()["ready"])
        self.assertEqual(lease.worker.engine.stops, 1)
        finish.set()
        await asyncio.gather(*manager.tasks, return_exceptions=True)
        self.assertFalse(lease.worker.active)
        self.assertFalse(manager.tasks)
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            await manager.shutdown()

    async def test_stop_timeout_is_reported_without_duplicate_cleanup(self):
        class StuckEngine(FakeEngine):
            async def stop(self):
                self.stops += 1
                await asyncio.Event().wait()

        manager = MultiWorkerManager(self.config("D"), engine_factory=StuckEngine)
        await manager.start()
        with patch("src.worker_manager.STOP_TIMEOUT", 0.01):
            with self.assertRaisesRegex(RuntimeError, "cleanup"):
                await asyncio.wait_for(manager.shutdown(), 1)
        self.assertTrue(all(w.state == "cleanup_failed" and w.engine.stops == 1
                            for w in manager.workers.values()))
        with self.assertRaisesRegex(RuntimeError, "cleanup"):
            await manager.shutdown()
        self.assertTrue(all(w.engine.stops == 1 for w in manager.workers.values()))

    async def test_metrics_report_configured_idle_gpu_and_lifecycle_evidence(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        metrics = manager.metrics()
        self.assertEqual(
            [row["physical_gpu_uuid"] for row in metrics["physical_gpus"]],
            ["GPU-00000000-0000-0000-0000-000000000001",
             "GPU-00000000-0000-0000-0000-000000000002"])
        self.assertTrue(all(row["telemetry"] == "unavailable" for row in metrics["physical_gpus"]))
        self.assertIsInstance(metrics["uptime_seconds"], float)
        self.assertTrue(all(isinstance(row["ready_seconds_after_start"], float)
                            for row in metrics["workers"].values()))

    async def test_metrics_deduplicate_physical_device_and_leave_unavailable_unknown(self):
        manager = MultiWorkerManager(self.config("B"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        metrics = manager.metrics()
        self.assertEqual(len(metrics["physical_gpus"]), 2)
        self.assertTrue(all(row["utilization_percent"] is None for row in metrics["physical_gpus"]))
        gpu0 = [row for row in metrics["physical_gpus"] if row["physical_gpu_uuid"].endswith("0001")]
        self.assertEqual(len(gpu0), 1)
        self.assertIn(LOW_MODEL, metrics["models"])
        self.assertNotIn("request_id", str(metrics))


    async def test_bounded_vllm_poll_keeps_unknown_and_stale_values_null(self):
        from src.measurements import parse_vllm_metrics

        self.assertEqual(parse_vllm_metrics(
            f'vllm:num_requests_running{{model_name="{LOW_MODEL}"}} 2\n'
            f'vllm:num_requests_running{{model_name="{LOW_MODEL}",engine_index="1"}} 3\n',
            LOW_MODEL), {})
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine, mode="remote")
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        calls = []

        def upstream(req):
            calls.append(req)
            worker = next(row for row in manager.config.workers if row.endpoint == str(req.url).removesuffix("/metrics"))
            self.assertEqual(req.headers["authorization"], "Bearer " + (
                "internal-small" if worker.model == LOW_MODEL else "internal-large"))
            model = worker.model
            return httpx.Response(200, text=(
                f'vllm:num_requests_running{{model_name="{model}"}} 2\n'
                f'vllm:num_requests_waiting{{model_name="{model}"}} 1\n'
                f'vllm:kv_cache_usage_perc{{model_name="{model}"}} 0.375\n'
                f'vllm:num_preemptions_total{{model_name="{model}"}} 4\n'))

        tokens = {row.service_token_env: (
            "internal-small" if row.model == LOW_MODEL else "internal-large")
            for row in manager.config.workers}
        with patch.dict("os.environ", tokens):
            async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
                await manager._poll_metrics(client)
        self.assertEqual(len(calls), 2)
        snapshot = manager.metrics()
        small = snapshot["workers"]["small-0"]["vllm"]
        self.assertEqual(small["num_requests_running"], 2)
        self.assertEqual(small["kv_cache_usage_fraction"], 0.375)
        self.assertEqual(small["num_preemptions"], 4)
        self.assertTrue(manager.readiness()["ready"])
        self.assertTrue(all(device["telemetry"] == "unavailable" for device in snapshot["physical_gpus"]))

        with patch.dict("os.environ", tokens):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda req: httpx.Response(200, content=b"x" * 262145))) as client:
                await manager._poll_metrics(client)
        self.assertEqual(manager.metrics()["workers"]["small-0"]["vllm"]["status"], "stale")
        self.assertIsNone(manager.metrics()["workers"]["small-0"]["vllm"]["num_requests_running"])
        self.assertTrue(manager.readiness()["ready"])
        manager.workers["small-0"].vllm_ok = True
        manager.workers["small-0"].vllm_sample_at -= 3 * manager.metrics_interval
        self.assertIsNone(manager.metrics()["workers"]["small-0"]["vllm"]["kv_cache_usage_fraction"])


if __name__ == "__main__":
    unittest.main()
