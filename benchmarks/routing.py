"""CPU-only, virtual-time policy comparison using the existing arrival generator and manager."""

import argparse
import asyncio
import heapq
import json
import random
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

from benchmarks.workload_generator.arrivals import arrivals
from src.model_router import RouterError, RoutingRequest
from src.serving_config import load_config
from src.vllm_engine import HIGH_MODEL, LOW_MODEL
from src.worker_manager import MultiWorkerManager

POLICIES = ("least-inflight", "round-robin", "least-queue-depth", "weighted-capacity")
SCENARIOS = ("balanced", "uneven-capacity", "saturation", "failure-draining", "concurrent", "changing-demand")
CONFIG = Path(__file__).resolve().parents[1] / "configs/serving-b.json"


class SimulatedEngine:
    def __init__(self, definition):
        self.model = definition.model

    async def prepare(self, model, messages, max_tokens):
        return {}


async def simulate(policy, scenario, seed=17):
    config = load_config(CONFIG)
    config = replace(config, policy=policy, workers=tuple(replace(w, capacity_weight=3)
        if scenario == "uneven-capacity" and w.id == "small-0" else w for w in config.workers))
    manager = MultiWorkerManager(config, engine_factory=SimulatedEngine)
    for worker in manager.workers.values():
        worker.state = "ready"
    rate = 20 if scenario == "saturation" else 6
    rng = random.Random(seed)
    schedule = list(arrivals(0, 6, rate, "poisson", rng))
    if scenario == "concurrent":
        schedule = [round(time, 1) for time in schedule]
    events = [(time, 1, index, None) for index, time in enumerate(schedule)]
    heapq.heapify(events)
    assignments = {worker.id: 0 for worker in config.workers}
    outcomes = {"reserved": 0, "server_busy": 0, "not_ready": 0, "simulated_released": 0}
    transitions = set()
    while events:
        time, kind, index, lease = heapq.heappop(events)
        if scenario == "failure-draining" and time >= 2 and "failed" not in transitions:
            manager.fail(exclude=SimpleNamespace(worker=manager.workers["small-0"]))
            transitions.add("failed")
        if scenario == "failure-draining" and time >= 4 and "draining" not in transitions:
            manager.workers["small-1"].state = "draining"
            transitions.add("draining")
        if kind == 0:
            if lease in lease.worker.active or lease in lease.worker.queue:
                manager.cancel_waiting(lease)
                outcomes["simulated_released"] += 1
            continue
        if scenario == "changing-demand":
            model = LOW_MODEL if time < 3 else HIGH_MODEL
        else:
            model = LOW_MODEL if index % 3 else HIGH_MODEL
        body = RoutingRequest(model=model, messages=[{"role": "user", "content": "simulated"}], max_tokens=1)
        try:
            lease = manager.submit(body, str(index))
        except RouterError as error:
            outcomes[error.code] += 1
            continue
        assert lease.worker.definition.model == model
        assert len(lease.worker.active) <= lease.worker.definition.max_sequences
        assert len(lease.worker.queue) <= manager.config.queue_capacity_per_worker
        outcomes["reserved"] += 1
        assignments[lease.worker.definition.id] += 1
        # Virtual service duration is an assumption, not observed GPU performance.
        duration = 1 / lease.worker.definition.capacity_weight if scenario == "uneven-capacity" else 1
        heapq.heappush(events, (round(time + duration, 9), 0, index, lease))
    await asyncio.gather(*manager.tasks, return_exceptions=True)
    return {"evidence": "simulation", "scenario": scenario, "seed": seed, "policy": policy,
            "assumptions": {"arrival_rate_rps": rate, "duration_s": 6,
                            "virtual_service_s": "1 / configured weight for uneven-capacity, otherwise 1",
                            "queue_capacity_per_worker": config.queue_capacity_per_worker},
            "outcomes": outcomes, "assignments": assignments}


def main():
    parser = argparse.ArgumentParser(description="Deterministic CPU-only simulated routing comparisons")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    async def compare():
        return [await simulate(policy, scenario, args.seed) for scenario in SCENARIOS for policy in POLICIES]

    print(json.dumps(asyncio.run(compare()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
