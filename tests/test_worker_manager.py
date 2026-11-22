import asyncio
import unittest
from pathlib import Path

from src.model_router import RouterError, RoutingRequest
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

    async def test_lost_dispatched_worker_recycles_only_its_replica(self):
        engines = {}
        def factory(row):
            engines[row.id] = FakeEngine(row)
            return engines[row.id]
        manager = MultiWorkerManager(self.config("B"), engine_factory=factory)
        self.addAsyncCleanup(manager.shutdown)
        await manager.start()
        lease = manager.submit(request(), "dispatched")
        await lease.ready
        other = manager.workers["large-0"]
        old_other_starts = other.engine.starts
        lease.dispatched = True
        manager.fail("worker_lost", exclude=lease)
        await manager.release(lease, False)
        self.assertEqual(lease.worker.generation, 1)
        self.assertEqual(lease.worker.state, "ready")
        self.assertEqual(other.engine.starts, old_other_starts)

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
