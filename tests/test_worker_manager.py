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
        self.assertTrue(state["ready"])
        self.assertTrue(manager.readiness()["ready"])
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

    async def test_cancelled_dispatch_recycles_worker_and_keeps_replica_routable(self):
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
        self.assertEqual(worker.state, "recycling")
        with self.assertRaises(RouterError) as caught:
            await queued.ready
        self.assertEqual(caught.exception.status, 503)
        await asyncio.gather(queued.prepare_task, return_exceptions=True)
        self.assertNotIn(queued, worker.queue)
        second.dispatched = True
        await manager.release(second, True)
        await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "ready")
        self.assertEqual(worker.engine.starts, 2)
        self.assertEqual(worker.engine.stops, 1)
        revived = manager.submit(request(), "after-recycle")
        await revived.ready
        self.assertIs(revived.worker, worker)
        await manager.release(revived, True)

    async def test_two_independent_cancellations_each_recover_the_same_worker(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        for name in ("first", "second"):
            lease = manager.submit(request(), name)
            await lease.ready
            self.assertIs(lease.worker, worker)
            lease.dispatched = True
            await manager.release(lease, False)
            self.assertEqual(worker.state, "recycling")
            await asyncio.gather(*manager.tasks)
            self.assertEqual(worker.state, "ready")
        self.assertEqual(worker.engine.starts, 3)
        self.assertEqual(worker.engine.stops, 2)
        served = manager.submit(request(), "after-recycles")
        await served.ready
        self.assertIs(served.worker, worker)
        await manager.release(served, True)

    async def test_failed_recycle_leaves_worker_unavailable(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        lease = manager.submit(request(), "dispatched")
        await lease.ready
        worker.engine.fail = True
        lease.dispatched = True
        with self.assertLogs("inference.workers", level="WARNING") as logs:
            await manager.release(lease, False)
            await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.last_error, "recycle_failed")
        self.assertIn("recycle_failed", "\n".join(logs.output))
        self.assertIn("worker small-0 worker_failed", "\n".join(logs.output))
        self.assertEqual(worker.engine.starts, 2)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "after-failure")
        self.assertEqual(caught.exception.code, "not_ready")

    async def test_quarantined_worker_leaves_its_model_replica_routable(self):
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
        await manager.release(lease, False)
        await asyncio.gather(*manager.tasks)
        self.assertEqual(failed.state, "ready")
        again = manager.submit(request(), "second")
        await again.ready
        self.assertIs(again.worker, failed)
        failed.engine.fail = True
        again.dispatched = True
        await manager.release(again, False)
        await asyncio.gather(*manager.tasks)
        self.assertEqual(failed.state, "unavailable")
        self.assertEqual(failed.engine.starts, 3)
        self.assertEqual(failed.engine.stops, 3)
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
        await manager._probe(dead, dead.generation)
        self.assertEqual(dead.state, "unavailable")
        self.assertEqual(manager.metrics()["workers"]["small-0"]["error"], "health_check_failed")
        lease = manager.submit(request(), "failover")
        await lease.ready
        self.assertEqual(lease.worker.definition.id, "small-1")
        await manager.release(lease, True)
        other = manager.workers["small-1"]
        other.engine.alive = False
        await manager._probe(other, other.generation)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "no-replica")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(caught.exception.code, "not_ready")
        state = manager.readiness()
        self.assertTrue(state["ready"])
        self.assertEqual(state["models"][LOW_MODEL]["status"], "unavailable")
        self.assertEqual(state["models"][HIGH_MODEL]["status"], "ready")
        large = manager.workers["large-0"]
        large.engine.alive = False
        await manager._probe(large, large.generation)
        self.assertFalse(manager.readiness()["ready"])

    async def test_recycling_worker_is_not_revived_by_concurrent_failure_or_probe(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        first = manager.submit(request(), "first")
        second = manager.submit(request(), "second")
        await asyncio.gather(first.ready, second.ready)
        worker.engine.fail = True
        first.dispatched = True
        await manager.release(first, False)
        self.assertEqual(worker.state, "recycling")
        self.assertIsNotNone(worker.cleanup_task)
        manager.fail("upstream_failed", exclude=second)
        self.assertEqual(worker.state, "recycling")
        served = {w.definition.endpoint: w.definition.model for w in manager.workers.values()}

        class HealthyUpstream:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                if url.endswith("/health"):
                    return httpx.Response(200)
                return httpx.Response(200, json={"data": [{"id": served[url[: -len("/v1/models")]]}]})

        with patch("src.worker_manager.httpx.AsyncClient", HealthyUpstream):
            await manager._probe(worker, worker.generation)
            await manager._probe(worker, worker.generation)
        self.assertEqual(worker.state, "recycling")
        second.dispatched = True
        await manager.release(second, True)
        await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.last_error, "recycle_failed")
        self.assertEqual(worker.engine.starts, 2)
        self.assertEqual(worker.engine.stops, 2)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "after-failure")
        self.assertEqual(caught.exception.code, "not_ready")

    async def test_probe_in_flight_cannot_quarantine_escape_after_recycle_starts(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        lease = manager.submit(request(), "dispatched")
        await lease.ready
        self.assertIs(lease.worker, worker)
        generation = worker.generation
        served = {w.definition.endpoint: w.definition.model for w in manager.workers.values()}
        probed = asyncio.Event()
        resumed = asyncio.Event()
        recycling = asyncio.Event()

        async def hold_recycle(manager_self, recycled_worker):
            recycling.set()
            await asyncio.Event().wait()

        class GatedUpstream:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                probed.set()
                await resumed.wait()
                if url.endswith("/health"):
                    return httpx.Response(503)
                return httpx.Response(200, json={"data": [{"id": served[url[: -len("/v1/models")]]}]})

        with patch.object(MultiWorkerManager, "_recycle", hold_recycle), \
                patch("src.worker_manager.httpx.AsyncClient", GatedUpstream):
            probe = asyncio.create_task(manager._probe(worker, generation))
            await asyncio.wait_for(probed.wait(), 2)
            lease.dispatched = True
            await manager.release(lease, False)
            self.assertEqual(worker.state, "recycling")
            self.assertIsNotNone(worker.cleanup_task)
            await asyncio.wait_for(recycling.wait(), 2)
            self.assertEqual(worker.generation, generation)
            resumed.set()
            await asyncio.wait_for(probe, 2)
        self.assertEqual(worker.state, "recycling")
        self.assertEqual(worker.engine.starts, 1)

    async def test_recycle_drain_covers_two_sequential_request_budgets(self):
        config = self.config("D")
        config = replace(config, workers=tuple(
            replace(row, request_timeout_s=1.0) for row in config.workers))
        manager = MultiWorkerManager(config, engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        with patch("src.worker_manager.DRAIN_TIMEOUT", 0.05):
            first = manager.submit(request(), "first")
            second = manager.submit(request(), "second")
            await asyncio.gather(first.ready, second.ready)
            first.dispatched = True
            await manager.release(first, False)
            self.assertEqual(worker.state, "recycling")
            await asyncio.sleep(1.5)
            self.assertEqual(worker.state, "recycling")
            second.dispatched = True
            await manager.release(second, True)
            await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "ready")
        self.assertIsNone(worker.last_error)
        self.assertEqual(worker.engine.starts, 2)
        self.assertEqual(worker.engine.stops, 1)
        served = manager.submit(request(), "after-recycle")
        await served.ready
        await manager.release(served, True)

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
        self.assertIsNone(worker.cleanup_task)
        self.assertEqual(worker.engine.starts, 1)
        self.assertEqual(worker.engine.stops, 0)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "after-death")
        self.assertEqual(caught.exception.status, 503)
        self.assertEqual(manager.readiness()["models"][LOW_MODEL]["status"], "unavailable")

    async def test_failed_recycle_quarantine_survives_later_lease_release(self):
        config = self.config("D")
        config = replace(config, workers=tuple(
            replace(row, request_timeout_s=0.01) for row in config.workers))
        manager = MultiWorkerManager(config, engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        first = manager.submit(request(), "first")
        second = manager.submit(request(), "second")
        await asyncio.gather(first.ready, second.ready)
        first.dispatched = True
        with patch("src.worker_manager.DRAIN_TIMEOUT", 0.01), \
                patch("src.worker_manager.CLOSE_TIMEOUT", 0.01):
            await manager.release(first, False)
            await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.last_error, "recycle_failed")
        self.assertEqual(worker.engine.starts, 1)
        second.dispatched = True
        await manager.release(second, False)
        self.assertIsNone(worker.cleanup_task)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.engine.starts, 1)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "after-failure")
        self.assertEqual(caught.exception.code, "not_ready")

    async def test_recycle_drain_covers_bounded_response_delivery(self):
        config = self.config("D")
        config = replace(config, workers=tuple(
            replace(row, request_timeout_s=0.05) for row in config.workers))
        manager = MultiWorkerManager(config, engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        first = manager.submit(request(), "first")
        second = manager.submit(request(), "second")
        await asyncio.gather(first.ready, second.ready)
        with patch("src.worker_manager.DRAIN_TIMEOUT", 0.01), \
                patch("src.worker_manager.CLOSE_TIMEOUT", 0.05), \
                patch("src.model_router.SEND_TIMEOUT", 0.2), \
                patch("src.model_router.CLOSE_TIMEOUT", 0.05):
            first.dispatched = True
            await manager.release(first, False)
            self.assertEqual(worker.state, "recycling")
            await asyncio.sleep(0.5)
            self.assertEqual(worker.state, "recycling")
            second.dispatched = True
            await manager.release(second, True)
            await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "ready")
        self.assertIsNone(worker.last_error)

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
        worker.state = "suspect"
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

    async def test_remote_dispatch_failure_quarantines_without_recycle(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine, mode="remote")
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        lease = manager.submit(request(), "remote-cancel")
        await lease.ready
        lease.dispatched = True
        await manager.release(lease, False)
        self.assertEqual(worker.state, "unavailable")
        self.assertIsNone(worker.cleanup_task)
        self.assertEqual(worker.engine.starts, 1)
        self.assertEqual(worker.engine.stops, 0)
        with self.assertRaises(RouterError) as caught:
            manager.submit(request(), "after-remote-cancel")
        self.assertEqual(caught.exception.status, 503)

    async def test_death_during_recycle_drain_is_never_restarted(self):
        manager = MultiWorkerManager(self.config("D"), engine_factory=FakeEngine)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        worker = next(w for w in manager.workers.values() if w.definition.model == LOW_MODEL)
        first = manager.submit(request(), "first")
        second = manager.submit(request(), "second")
        await asyncio.gather(first.ready, second.ready)
        with patch("src.worker_manager.DRAIN_TIMEOUT", 0.05):
            first.dispatched = True
            await manager.release(first, False)
            self.assertEqual(worker.state, "recycling")
            worker.engine.alive = False
            second.dispatched = True
            await manager.release(second, False)
            await asyncio.gather(*manager.tasks)
        self.assertEqual(worker.state, "unavailable")
        self.assertEqual(worker.last_error, "process_exited")
        self.assertEqual(worker.engine.starts, 1)
        self.assertEqual(worker.engine.stops, 0)
        self.assertIsNone(worker.cleanup_task)

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


if __name__ == "__main__":
    unittest.main()
