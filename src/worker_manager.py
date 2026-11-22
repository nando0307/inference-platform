"""Event-loop-owned registry and bounded dispatcher for independent GPU workers."""

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from time import perf_counter

import httpx

from .inference import ContextLimitError
from .model_router import CLOSE_TIMEOUT, DRAIN_TIMEOUT, Lease, RouterError, STOP_TIMEOUT, _observe
from .serving_config import ServingConfig
from .vllm_engine import VLLMEngine

_nvml_initialized = False

logger = logging.getLogger("inference.workers")


@dataclass(eq=False)
class Worker:
    definition: object
    engine: VLLMEngine
    state: str = "starting"
    generation: int = 0
    active: set = field(default_factory=set)
    queue: deque = field(default_factory=deque)
    available: asyncio.Event = field(default_factory=asyncio.Event)
    consecutive_successes: int = 0
    requests: dict = field(default_factory=lambda: {"completed": 0, "failed": 0, "canceled": 0, "rejected": 0})
    last_error: str | None = None
    cleanup_task: asyncio.Task | None = None

    @property
    def inflight(self):
        return len(self.active) + len(self.queue)


class RemoteWorkerEngine:
    """Authenticated client for a separately launched worker on another host."""

    def __init__(self, definition):
        self.worker = definition
        self.base_url = definition.endpoint.rstrip("/")
        self.model = None
        self.process = None
        self._tokenizer = VLLMEngine(worker=definition)

    def is_alive(self):
        return self.model is not None

    async def prepare(self, model, messages, max_tokens):
        return await self._tokenizer.prepare(model, messages, max_tokens)

    async def start(self, model):
        token = os.environ.get(self.worker.service_token_env)
        if not token:
            raise RuntimeError("Worker service credential is not configured.")
        headers = {"authorization": f"Bearer {token}"}
        async with asyncio.timeout(self.worker.startup_timeout_s):
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                while True:
                    try:
                        health = await client.get(self.base_url + "/health", headers=headers)
                        models = await client.get(self.base_url + "/v1/models", headers=headers)
                        if (health.status_code == 200 and models.status_code == 200
                                and [row["id"] for row in models.json()["data"]] == [model]):
                            self.model = model
                            return
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.1)

    async def stop(self):
        self.model = None


class MultiWorkerManager:
    """Static workers, exact-model dispatch and per-worker bounded waiting."""
    def __init__(self, config: ServingConfig, *, engine_factory=None, mode="local"):
        if mode not in {"local", "remote"}:
            raise ValueError("Worker mode must be local or remote.")
        if mode == "local" and any(row.host != "local" for row in config.workers):
            raise ValueError("Local worker management requires host ID 'local'.")
        self.config = config
        self.mode = mode
        if engine_factory is None:
            engine_factory = (lambda definition: VLLMEngine(worker=definition)) if mode == "local" else RemoteWorkerEngine
        self.workers = {row.id: Worker(row, engine_factory(row)) for row in config.workers}
        self.accepting = True
        self.closing = False
        self.tasks = set()
        self.health_task = None
        self.cleanup_error = None
        self.events = deque(maxlen=2048)

    def emit(self, event, worker=None, **fields):
        self.events.append({"timestamp": perf_counter(), "event": event,
                            "worker": worker.definition.id if worker else None, **fields})
        if event in {"startup_failed", "unready", "worker_failed"}:
            logger.warning("worker %s %s %s", worker.definition.id if worker else None,
                           event, " ".join(f"{key}={value}" for key, value in fields.items()))

    def snapshot(self):
        models = {}
        for model in self.config.models.values():
            replicas = [w for w in self.workers.values() if w.definition.model == model.id]
            ready = sum(w.state == "ready" for w in replicas)
            models[model.id] = {"ready": ready, "configured": len(replicas),
                                "status": "ready" if ready == len(replicas) else "degraded" if ready else "unavailable"}
        return {
            "ready": self.accepting and all(row["ready"] > 0 for row in models.values()),
            "placement": self.config.placement, "models": models,
            "workers": {w.definition.id: {
                "state": w.state, "generation": w.generation, "model": w.definition.model,
                "revision": self._revision(w.definition.model), "host": w.definition.host,
                "physical_gpu_uuid": w.definition.gpu_uuid, "cuda_device": w.definition.cuda_device,
                "in_flight": len(w.active), "queued": len(w.queue),
                "queue_capacity": self.config.queue_capacity_per_worker, "error": w.last_error,
            } for w in self.workers.values()},
        }

    def readiness(self):
        state = self.snapshot()
        return {"ready": state["ready"], "models": state["models"]}

    def _revision(self, model_id):
        return next(model.revision for model in self.config.models.values() if model.id == model_id)

    async def start(self):
        async def start_worker(worker, generation):
            try:
                async with asyncio.timeout(worker.definition.startup_timeout_s):
                    await worker.engine.start(worker.definition.model)
                if generation == worker.generation:
                    if worker.engine.model != worker.definition.model or not worker.engine.is_alive():
                        raise RuntimeError("Worker did not serve the configured model.")
                    worker.state = "ready"
                    worker.last_error = None
                    self.emit("ready", worker, generation=generation)
            except Exception as error:
                if generation == worker.generation:
                    worker.state = "unavailable"
                    worker.last_error = type(error).__name__
                    self.emit("startup_failed", worker, generation=generation)
                    try:
                        await worker.engine.stop()
                    except Exception:
                        worker.last_error = "cleanup_failed"
            finally:
                if worker.state == "starting":
                    worker.state = "unavailable"

        await asyncio.gather(*(start_worker(w, w.generation) for w in self.workers.values()))
        self.health_task = asyncio.create_task(self._health_loop(), name="multi-gpu-worker-health")
        return any(w.state == "ready" for w in self.workers.values())

    async def _probe(self, worker, generation):
        if worker.generation != generation or worker.state not in {"ready", "suspect"}:
            return
        alive = self.mode != "local" or worker.engine.is_alive()
        if not alive:
            valid = False
        else:
            token = os.environ.get(worker.definition.service_token_env, "")
            headers = {"authorization": f"Bearer {token}"} if token else {}
            try:
                async with httpx.AsyncClient(timeout=self.config.health_timeout_s, trust_env=False) as client:
                    health = await client.get(worker.engine.base_url + "/health", headers=headers)
                    models = await client.get(worker.engine.base_url + "/v1/models", headers=headers)
                    valid = (health.status_code == 200 and models.status_code == 200
                             and [item["id"] for item in models.json()["data"]] == [worker.definition.model])
            except Exception:
                valid = False
        if worker.generation != generation:
            return
        if valid:
            worker.consecutive_successes += 1
            if worker.state == "suspect" and worker.consecutive_successes >= 2:
                worker.state = "ready"
                worker.last_error = None
                self.emit("recovered", worker, generation=generation)
        else:
            worker.state = "suspect" if alive else "unavailable"
            worker.consecutive_successes = 0
            worker.last_error = "health_check_failed"
            self.emit("unready", worker, generation=generation)
            self._fail_queued(worker, "not_ready")

    async def _health_loop(self):
        try:
            while not self.closing:
                await asyncio.sleep(self.config.health_interval_s)
                await asyncio.gather(*(self._probe(w, w.generation) for w in self.workers.values()))
        except asyncio.CancelledError:
            pass

    def submit(self, body, request_id):
        if not self.accepting:
            raise RouterError()
        candidates = sorted((w for w in self.workers.values()
            if w.definition.model == body.model and w.state == "ready"
            and (len(w.active) < w.definition.max_sequences
                 or len(w.queue) < self.config.queue_capacity_per_worker)),
            key=lambda w: (w.inflight, w.definition.id))
        if not candidates:
            ready = [w for w in self.workers.values()
                     if w.definition.model == body.model and w.state == "ready"]
            if ready:
                rejected = min(ready, key=lambda w: (w.inflight, w.definition.id))
                rejected.requests["rejected"] += 1
                raise RouterError(429, "server_busy", "All replicas for the requested model are saturated.")
            raise RouterError(503, "not_ready", "The requested model is unavailable.")
        worker = candidates[0]
        lease = Lease(request_id, body, asyncio.get_running_loop().create_future())
        lease.ready.add_done_callback(_observe)
        lease.worker, lease.generation = worker, worker.generation
        if len(worker.active) < worker.definition.max_sequences:
            worker.active.add(lease)
        else:
            worker.queue.append(lease)
        self.emit("queued", worker, request_id=request_id)
        task = asyncio.create_task(self._prepare(lease), name=f"prepare-{worker.definition.id}")
        lease.prepare_task = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return lease

    async def _prepare(self, lease):
        worker = lease.worker
        try:
            lease.prepared = await worker.engine.prepare(
                lease.request.model, [message.model_dump() for message in lease.request.messages],
                lease.request.max_tokens,
            )
            while lease in worker.queue:
                worker.available.clear()
                if lease not in worker.queue:
                    break
                await worker.available.wait()
            if worker.generation != lease.generation or worker.state != "ready":
                raise RouterError(503, "not_ready", "The selected worker became unavailable.")
            if lease not in worker.active:
                raise RouterError(503, "not_ready", "The selected worker lost its admission slot.")
            lease.ready.set_result(lease)
            self.emit("admitted", worker, request_id=lease.request_id)
        except asyncio.CancelledError:
            self.cancel_waiting(lease)
            raise
        except Exception as error:
            if lease in worker.queue:
                worker.queue.remove(lease)
            if lease in worker.active:
                worker.active.remove(lease)
                self._promote(worker)
            if not lease.ready.done():
                if isinstance(error, RouterError):
                    public = error
                elif isinstance(error, ContextLimitError):
                    public = RouterError(400, "context_length_exceeded", "Prompt and output exceed the supported context.")
                else:
                    public = RouterError(503, "engine_failed", "The selected worker could not prepare the request.")
                    worker.state = "suspect"
                    worker.last_error = type(error).__name__
                lease.ready.set_exception(public)
            worker.requests["failed"] += 1
            worker.available.set()

    def cancel_waiting(self, lease):
        worker = getattr(lease, "worker", None)
        if worker is None or lease.dispatched:
            return
        removed = False
        if lease in worker.queue:
            worker.queue.remove(lease)
            removed = True
        if lease in worker.active:
            worker.active.remove(lease)
            removed = True
            self._promote(worker)
        if removed:
            task = getattr(lease, "prepare_task", None)
            if task is not None and task is not asyncio.current_task() and not task.done():
                task.cancel()
            if not lease.ready.done():
                lease.ready.cancel()
            worker.requests["canceled"] += 1
            self.emit("canceled", worker, request_id=lease.request_id)
            worker.available.set()

    def _promote(self, worker):
        while worker.queue and len(worker.active) < worker.definition.max_sequences:
            lease = worker.queue.popleft()
            worker.active.add(lease)
        worker.available.set()

    def _fail_queued(self, worker, code):
        failed = False
        while worker.queue:
            lease = worker.queue.popleft()
            failed = True
            if not lease.ready.done():
                lease.ready.set_exception(RouterError(503, code, "The selected worker became unavailable."))
            task = lease.prepare_task
            if task is not None and task is not asyncio.current_task() and not task.done():
                task.cancel()
        if failed:
            worker.available.set()

    async def release(self, lease, completed):
        worker = lease.worker
        self.cancel_waiting(lease)
        if lease not in worker.active:
            return
        if lease.upstream is not None:
            try:
                async with asyncio.timeout(CLOSE_TIMEOUT):
                    await lease.upstream.aclose()
            except Exception:
                completed = False
                worker.state = "suspect"
                worker.last_error = "upstream_close_failed"
        worker.active.remove(lease)
        outcome = "completed" if completed else "canceled" if lease.task and lease.task.cancelled() else "failed"
        worker.requests[outcome] += 1
        self.emit("released", worker, request_id=lease.request_id, outcome=outcome)
        if not completed and lease.dispatched and not self.closing:
            worker.state = "unavailable"
            worker.last_error = outcome
            self._fail_queued(worker, "not_ready")
            if worker.cleanup_task is None:
                worker.cleanup_task = asyncio.create_task(self._retire(worker))
                self.tasks.add(worker.cleanup_task)
                worker.cleanup_task.add_done_callback(self.tasks.discard)
        self._promote(worker)

    def fail(self, reason="upstream_failed", exclude=None):
        worker = getattr(exclude, "worker", None) if exclude is not None else None
        if worker is None:
            return
        worker.state = "suspect"
        worker.last_error = reason
        worker.consecutive_successes = 0
        self.emit("worker_failed", worker, reason=reason, generation=worker.generation)
        self._fail_queued(worker, "not_ready")

    async def _retire(self, worker):
        worker.generation += 1
        try:
            async with asyncio.timeout(DRAIN_TIMEOUT):
                while worker.active:
                    await asyncio.sleep(0.01)
        except TimeoutError:
            pass
        try:
            async with asyncio.timeout(STOP_TIMEOUT):
                await worker.engine.stop()
        except Exception:
            worker.last_error = "cleanup_failed"

    def metrics(self):
        workers = self.snapshot()["workers"]
        worker_counts = {w.definition.id: dict(w.requests) for w in self.workers.values()}
        models = {}
        for model in self.config.models.values():
            members = [w for w in self.workers.values() if w.definition.model == model.id]
            models[model.id] = {"ready": sum(w.state == "ready" for w in members),
                "requests": {key: sum(w.requests[key] for w in members)
                             for key in ("completed", "failed", "canceled", "rejected")}}
        devices = {}
        for worker in self.workers.values():
            key = (worker.definition.host, worker.definition.gpu_uuid)
            if key in devices:
                continue
            sample = self._nvml_sample(key[1]) if self.mode == "local" and key[0] == "local" else None
            devices[key] = {"host": key[0], "physical_gpu_uuid": key[1],
                "utilization_percent": None, "memory_used_bytes": None,
                "memory_total_bytes": None, "telemetry": "unavailable"}
            if sample is not None:
                devices[key].update(sample)
        return {"workers": workers, "worker_requests": worker_counts,
                "models": models, "physical_gpus": list(devices.values())}

    def _nvml_sample(self, gpu_uuid):
        global _nvml_initialized
        try:
            import pynvml
            if not _nvml_initialized:
                pynvml.nvmlInit()
                _nvml_initialized = True
            handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode())
            usage = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            return {"name": name, "utilization_percent": usage.gpu,
                    "memory_used_bytes": memory.used, "memory_total_bytes": memory.total,
                    "telemetry": "available"}
        except Exception:
            return None

    async def shutdown(self):
        self.accepting = False
        self.closing = True
        if self.health_task:
            self.health_task.cancel()
            await asyncio.gather(self.health_task, return_exceptions=True)
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for worker in self.workers.values():
            self._fail_queued(worker, "not_ready")
            for lease in list(worker.active):
                if lease.task and lease.task is not asyncio.current_task():
                    lease.task.cancel()
        try:
            async with asyncio.timeout(DRAIN_TIMEOUT):
                while any(w.active for w in self.workers.values()):
                    await asyncio.sleep(0.01)
        except TimeoutError:
            self.cleanup_error = True
        async def stop(worker):
            try:
                async with asyncio.timeout(STOP_TIMEOUT):
                    await worker.engine.stop()
                worker.state = "stopped"
            except Exception:
                worker.state = "cleanup_failed"
                self.cleanup_error = True
        await asyncio.gather(*(stop(worker) for worker in self.workers.values()))
        if self.cleanup_error:
            raise RuntimeError("At least one worker process group failed cleanup.")
