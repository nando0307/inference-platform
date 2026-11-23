"""Validated static worker placement configuration for the opt-in GPU router."""

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .vllm_engine import HIGH_MODEL, LOW_MODEL, MODEL_SPECS

SCHEMA_VERSION = 1
MAX_CONTEXT_TOKENS = 2048
MAX_SEQUENCES = 2
MAX_QUEUE_CAPACITY = 2
_GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F-]{36}$")


@dataclass(frozen=True)
class Model:
    id: str
    revision: str
    context_tokens: int


@dataclass(frozen=True)
class Host:
    id: str
    gpu_uuids: tuple[str, ...]


@dataclass(frozen=True)
class Worker:
    id: str
    model: str
    host: str
    cuda_device: int
    gpu_uuid: str
    endpoint: str
    memory_utilization: float
    max_sequences: int
    startup_timeout_s: float
    request_timeout_s: float
    service_token_env: str


@dataclass(frozen=True)
class ServingConfig:
    schema_version: int
    placement: str
    models: dict[str, Model]
    hosts: dict[str, Host]
    workers: tuple[Worker, ...]
    queue_capacity_per_worker: int
    health_interval_s: float
    health_timeout_s: float


def _positive(value, name, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be in (0, {maximum}]." if maximum else f"{name} must be positive.")
    return value


def validate_config(raw: dict) -> ServingConfig:
    """Validate all placement invariants before a worker/process can be created."""
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "placement", "models", "hosts", "workers", "routing"}:
        raise ValueError("Serving configuration has unsupported schema or fields.")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported serving configuration schema.")
    placement = raw.get("placement")
    if placement not in {"A", "B", "C", "D"}:
        raise ValueError("placement must be A, B, C, or D.")

    models_raw = raw.get("models")
    if not isinstance(models_raw, dict) or set(models_raw) != {"small", "large"}:
        raise ValueError("models must define exactly small and large.")
    models = {}
    expected_ids = {"small": LOW_MODEL, "large": HIGH_MODEL}
    for name, model in models_raw.items():
        if not isinstance(model, dict) or set(model) != {"id", "revision", "dtype", "context_tokens"} or model.get("id") != expected_ids[name]:
            raise ValueError(f"{name} must use its pinned allowlisted model and schema.")
        revision = model.get("revision")
        if revision != MODEL_SPECS[expected_ids[name]]["revision"]:
            raise ValueError(f"{name} must use the immutable repository revision.")
        context = model.get("context_tokens")
        if type(context) is not int or context != MAX_CONTEXT_TOKENS:
            raise ValueError("context_tokens must be the supported 2048-token policy.")
        if model.get("dtype") != "float16":
            raise ValueError("dtype must be float16.")
        models[name] = Model(expected_ids[name], revision, context)

    hosts_raw = raw.get("hosts")
    if not isinstance(hosts_raw, dict) or not hosts_raw:
        raise ValueError("At least one host is required.")
    hosts, global_uuids = {}, set()
    for host_id, host in hosts_raw.items():
        uuids = host.get("gpu_uuids") if isinstance(host, dict) and set(host) == {"gpu_uuids"} else None
        if (not isinstance(host_id, str) or not host_id or not isinstance(uuids, list)
                or not uuids or any(not isinstance(uuid, str) or not _GPU_UUID.fullmatch(uuid) for uuid in uuids)):
            raise ValueError("Each host requires a nonempty list of physical GPU UUIDs.")
        if len(set(uuids)) != len(uuids) or global_uuids.intersection(uuids):
            raise ValueError("Physical GPU UUIDs must be unique within this static topology.")
        global_uuids.update(uuids)
        hosts[host_id] = Host(host_id, tuple(uuids))

    workers_raw = raw.get("workers")
    if not isinstance(workers_raw, list) or not workers_raw:
        raise ValueError("At least one worker is required.")
    workers, ids, origins, slots, token_envs = [], set(), set(), set(), set()
    for row in workers_raw:
        if not isinstance(row, dict) or set(row) != set(Worker.__dataclass_fields__):
            raise ValueError("Each worker must use the complete supported worker schema.")
        worker = Worker(**row)
        if not isinstance(worker.id, str) or not worker.id or worker.id in ids:
            raise ValueError("Worker IDs must be nonempty strings and unique.")
        ids.add(worker.id)
        if not isinstance(worker.model, str) or worker.model not in {model.id for model in models.values()}:
            raise ValueError("Worker references an unknown model.")
        if (not isinstance(worker.host, str) or worker.host not in hosts
                or type(worker.cuda_device) is not int
                or not 0 <= worker.cuda_device < len(hosts[worker.host].gpu_uuids)):
            raise ValueError("Worker references an unknown host or local CUDA ordinal.")
        if not isinstance(worker.gpu_uuid, str) or hosts[worker.host].gpu_uuids[worker.cuda_device] != worker.gpu_uuid:
            raise ValueError("Worker CUDA ordinal and physical GPU UUID disagree.")
        if worker.gpu_uuid not in global_uuids:
            raise ValueError("Worker references an unknown physical GPU.")
        slot = (worker.host, worker.gpu_uuid, worker.model)
        if slot in slots:
            raise ValueError("A model may have only one worker per physical GPU.")
        slots.add(slot)
        if not isinstance(worker.endpoint, str):
            raise ValueError("Worker endpoint must be a string.")
        parsed = urlsplit(worker.endpoint)
        if (parsed.scheme != "http" or not parsed.hostname or not parsed.port
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("Worker endpoint must be an explicit credential-free HTTP origin with a port.")
        origin = (parsed.scheme, "127.0.0.1" if parsed.hostname == "localhost" else parsed.hostname,
                  parsed.port)
        if origin in origins:
            raise ValueError("Worker endpoints must be unique.")
        origins.add(origin)
        if (not isinstance(worker.service_token_env, str) or not worker.service_token_env.isidentifier()
                or worker.service_token_env in token_envs):
            raise ValueError("Each worker requires a distinct service-token environment variable name.")
        token_envs.add(worker.service_token_env)
        _positive(worker.memory_utilization, "memory_utilization", maximum=0.9)
        if type(worker.max_sequences) is not int or worker.max_sequences != MAX_SEQUENCES:
            raise ValueError("max_sequences must be 2.")
        _positive(worker.startup_timeout_s, "startup_timeout_s", maximum=360)
        _positive(worker.request_timeout_s, "request_timeout_s", maximum=180)
        workers.append(worker)

    if {worker.model for worker in workers} != {model.id for model in models.values()}:
        raise ValueError("Each configured model must have at least one worker.")
    expected = {
        "A": {("small", 0), ("large", 1)},
        "B": {("small", 0), ("large", 0), ("small", 1)},
        "C": {("small", 0), ("large", 0), ("large", 1)},
        "D": {("small", 0), ("large", 0)},
    }[placement]
    expected_ids = {
        "A": {"small-0", "large-1"},
        "B": {"small-0", "large-0", "small-1"},
        "C": {"small-0", "large-0", "large-1"},
        "D": {"small-0", "large-0"},
    }[placement]
    if len(hosts) != 1 or len(hosts[next(iter(hosts))].gpu_uuids) != 2:
        raise ValueError("A-D placements require one host with exactly two physical GPUs.")
    host_id = next(iter(hosts))
    actual = {("small" if worker.model == LOW_MODEL else "large", worker.cuda_device)
              for worker in workers if worker.host == host_id}
    if actual != expected or len(workers) != len(expected) or {worker.id for worker in workers} != expected_ids:
        raise ValueError(f"Worker topology does not match placement {placement}.")
    per_gpu_budget = {}
    for worker in workers:
        key = (worker.host, worker.gpu_uuid)
        per_gpu_budget[key] = per_gpu_budget.get(key, 0.0) + worker.memory_utilization
    if any(value > 1.0 for value in per_gpu_budget.values()):
        raise ValueError("Declared per-process memory budgets exceed the per-GPU policy cap.")

    routing = raw.get("routing")
    if not isinstance(routing, dict) or set(routing) != {
        "policy", "queue_capacity_per_worker", "health_interval_s", "health_timeout_s",
    } or routing.get("policy") != "least-inflight":
        raise ValueError("routing must match the supported least-inflight schema.")
    if routing.get("queue_capacity_per_worker") != MAX_QUEUE_CAPACITY:
        raise ValueError("queue_capacity_per_worker must be 2.")
    interval = _positive(routing.get("health_interval_s"), "health_interval_s", maximum=60)
    timeout = _positive(routing.get("health_timeout_s"), "health_timeout_s", maximum=30)
    if timeout > interval:
        raise ValueError("health timeout may not exceed the health interval.")
    return ServingConfig(SCHEMA_VERSION, placement, models, hosts, tuple(workers), MAX_QUEUE_CAPACITY, interval, timeout)


def load_config(path: str | Path) -> ServingConfig:
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(json.load(stream))


def dry_run(config: ServingConfig) -> list[dict]:
    """Return safe, deterministic host/UUID/CUDA mapping without allocating workers."""
    return [{"worker": worker.id, "model": worker.model, "host": worker.host,
             "physical_gpu_uuid": worker.gpu_uuid, "cuda_device": worker.cuda_device,
             "endpoint": worker.endpoint}
            for worker in sorted(config.workers, key=lambda item: item.id)]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate and print the static worker/GPU mapping.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps({"placement": config.placement, "workers": dry_run(config)}, indent=2))


if __name__ == "__main__":
    main()
