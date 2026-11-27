"""CPU-only Python baseline for the Rust bench binary; writes raw JSON lines to stdout."""
import asyncio
import json
import sys
from pathlib import Path
from time import perf_counter_ns
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model_router import RouterError
from src.routing import Candidate, select
from src.serving_config import load_config
from src.vllm_engine import LOW_MODEL
from src.worker_manager import MultiWorkerManager


class Engine:
    async def prepare(self, model, messages, max_tokens):
        return {}


def manager():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/serving-b.json")
    m = MultiWorkerManager(config, engine_factory=lambda _: Engine())
    for w in m.workers.values():
        w.state = "ready"
    return m


async def sample(kind, iterations):
    m = manager() if kind != "selector" else None
    body = SimpleNamespace(model=LOW_MODEL, messages=[], max_tokens=1)
    candidates = (Candidate("small-0", 0, 0, 1), Candidate("small-1", 1, 0, 1))
    samples, rejected, capacity = [], 0, 0
    wall = perf_counter_ns()
    for i in range(iterations):
        start = perf_counter_ns()
        if kind == "selector":
            select(candidates, "least-inflight", None, {})
        elif kind == "reservation":
            lease = m.submit(body, str(i))
            samples.append(perf_counter_ns() - start)
            m.cancel_waiting(lease)
            # Drain cancellation outside the measured selection/reservation interval.
            if i % 100 == 99:
                await asyncio.sleep(0)
            continue
        elif kind == "synthetic":
            leases = []
            for j in range(9):
                try:
                    leases.append(m.submit(body, f"{i}-{j}"))
                except RouterError as e:
                    assert e.code == "server_busy"
                    rejected += 1
            capacity = max(capacity, len(leases))
            for lease in leases:
                await lease.ready
                await m.release(lease, True)
        else:
            raise ValueError(kind)
        samples.append(perf_counter_ns() - start)
    await asyncio.gather(*m.tasks, return_exceptions=True) if m is not None else asyncio.sleep(0)
    return dict(samples_ns=samples, elapsed_ns=perf_counter_ns() - wall, rejected=rejected, capacity=capacity)


def main():
    kind, warmup, iterations, repeats = sys.argv[1], *map(int, sys.argv[2:])
    asyncio.run(sample(kind, warmup))
    for repeat in range(repeats):
        print(json.dumps(dict(workload=kind, repeat=repeat, **asyncio.run(sample(kind, iterations)))))


if __name__ == "__main__":
    main()
