"""CPU-only, causal discrete-event comparison of static A-D and dry-run adaptive placement."""

import argparse
from dataclasses import asdict, replace
import heapq
import json
from pathlib import Path

from benchmarks.workload_generator.config import load_scenario
from benchmarks.workload_generator.trace import generate_trace
from src.placement_planner import (CONSERVATIVE, SHORT_EXPERIMENT, History, Snapshot, plan)
from src.serving_config import load_config

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
# Explicitly assumed illustrative per-worker SLO-compliant service rates and footprints.
ASSUMED_RPS = {"small": .7, "large": .45}
ASSUMED_MEMORY = {"small": .35, "large": .40}
SHARED_GPU_FACTOR = .7
SLO_S = 10.0
MAX_ASSUMED_DEMAND_RPS = 100.0
SCENARIOS = ("balanced", "small_peak", "large_peak", "alternating_peaks", "sudden_spike",
             "microburst", "overload", "idle", "mixed_15min")


def configurations():
    return {name: load_config(ROOT / "configs" / f"serving-{name.lower()}.json") for name in "ABCD"}


def assumptions(configs):
    workers = {w.id: w for config in configs.values() for w in config.workers}
    models = {name.id: key for key, name in configs["A"].models.items()}
    rates = {id: ASSUMED_RPS[models[w.model]] for id, w in workers.items()}
    return workers, models, rates


def run(scenario, configs, initial="A", profile=SHORT_EXPERIMENT, *, adaptive=True,
        missing_windows=(), stale_windows=(), records=None):
    """Arrival, completion, transition and evaluation events; no planner access to future arrivals."""
    if initial not in configs:
        raise ValueError("Unknown initial placement")
    records = generate_trace(scenario) if records is None else records
    workers, models, capacity = assumptions(configs)
    memory = {id: ASSUMED_MEMORY[models[w.model]] for id, w in workers.items()}
    active = set(w.id for w in configs[initial].workers)
    current = initial
    history = History()
    window = {model: 0 for model in models}
    events = []
    sequence = 0
    for row in records:
        t = row["scheduled_offset_s"]
        if t < scenario.duration_s:
            heapq.heappush(events, (t, 2, sequence, "arrival", row))
            sequence += 1
    t = profile.interval_s
    while t <= scenario.duration_s + 1e-8:
        heapq.heappush(events, (round(t, 9), 1, sequence, "evaluate", None))
        sequence += 1
        t += profile.interval_s
    slots = {id: [] for id in workers}  # active service completion times, two slots per worker
    pending = None
    recommendations = []
    offered = completed = rejected = late = 0
    impacts = 0.0
    switches = 0
    while events:
        at, _, _, kind, value = heapq.heappop(events)
        if kind == "arrival":
            model = value["model"]
            window[model] += 1
            offered += 1
            choices = []
            for id in sorted(active):
                if workers[id].model != model:
                    continue
                available = slots[id]
                # At most two in service, two waiting (matches serving_config).
                if len(available) >= 4:
                    continue
                colocated = sum(workers[peer].gpu_uuid == workers[id].gpu_uuid for peer in active) > 1
                service = 1 / (capacity[id] * (SHARED_GPU_FACTOR if colocated else 1))
                start = at if len(available) < 2 else sorted(available)[-2]
                finish = max(at, start) + service
                choices.append((finish, id))
            if not choices:
                rejected += 1
                continue
            finish, id = min(choices)
            slots[id].append(finish)
            heapq.heappush(events, (finish, 0, sequence, "complete", (id, at, value)))
            sequence += 1
        elif kind == "complete":
            id, arrived, row = value
            slots[id].remove(at)
            if at - arrived <= min(SLO_S, row["request_timeout_s"]) and row["cancel_after_s"] is None:
                completed += 1
            else:
                late += 1
        elif kind == "step":
            step, rest, target = value
            # Removal waits for already-admitted work to drain; admissions remain
            # open until the atomic step boundary. No requests are killed.
            if step.action == "remove" and any(end > at for end in slots[step.worker]):
                next_at = max(at, max(slots[step.worker]))
                heapq.heappush(events, (next_at, 0, sequence, "step", value))
                sequence += 1
                continue
            if step.action == "remove":
                active.remove(step.worker)
            else:
                active.add(step.worker)
            if rest:
                heapq.heappush(events, (at + rest[0].impact_s, 0, sequence, "step", (rest[0], rest[1:], target)))
                sequence += 1
            else:
                current = target
                history = replace(history, last_change_s=at)
                pending = None
        else:
            index = round(at / profile.interval_s)
            if adaptive:
                observed = {model: window[model] / profile.interval_s for model in models}
                stamp = None if index in missing_windows else max(0, at - profile.max_age_s - 1) if index in stale_windows else at
                snapshot = Snapshot(SCHEMA_VERSION, at, stamp, observed,
                                    {model: MAX_ASSUMED_DEMAND_RPS for model in models},
                                    at, capacity,
                                    "assumed", {id: True for id in workers},
                                    memory, at, "assumed", shared_gpu_factor=SHARED_GPU_FACTOR)
                if pending is None:
                    decision = plan(configs, current, snapshot, history, profile)
                    history = decision.history
                    recommendations.append(decision.to_dict())
                    if decision.action == "recommend":
                        switches += 1
                        impacts += decision.transition_lost_requests
                        pending = decision.candidate
                        first, *rest = decision.transition
                        heapq.heappush(events, (at + first.impact_s, 0, sequence, "step", (first, tuple(rest), pending)))
                        sequence += 1
            window = {model: 0 for model in models}
    return {"schema_version": SCHEMA_VERSION, "kind": "simulation", "scenario": scenario.id,
            "seed": scenario.seed, "policy": "adaptive" if adaptive else f"static-{initial}",
            "initial": initial, "final": current, "profile": asdict(profile), "assumptions": {
                "capacity_provenance": "assumed", "memory_provenance": "assumed",
                "dedicated_worker_rps": capacity, "shared_gpu_factor": SHARED_GPU_FACTOR,
                "worker_memory_fraction": memory, "demand_limit_rps": MAX_ASSUMED_DEMAND_RPS,
                "slo_s": SLO_S, "service_model": "two parallel slots per worker, two waiting; constant seconds/request = 1/assumed_rps",
                "measurement_s": scenario.duration_s, "warmup_s": 0},
            "offered": offered, "simulated_slo_goodput": completed,
            "simulated_unmet_demand": offered - completed, "rejected": rejected, "late_or_canceled": late,
            "transitions": switches, "estimated_transition_lost_requests": impacts,
            "recommendations": recommendations}


def compare(scenario, configs=None, profile=SHORT_EXPERIMENT, **kwargs):
    configs = configs or configurations()
    trace = generate_trace(scenario)
    return {"schema_version": SCHEMA_VERSION, "kind": "simulation-comparison",
            "scenario": scenario.id, "runs": [run(scenario, configs, name, profile, adaptive=False, records=trace) for name in "ABCD"]
            + [run(scenario, configs, "A", profile, records=trace, **kwargs)]}


def sensitivity(configs=None, scenarios=("small_peak", "large_peak", "alternating_peaks", "mixed_15min")):
    configs = configs or configurations()
    output = []
    for name in scenarios:
        scenario = load_scenario(ROOT / "configs" / "workloads" / f"{name}.json")
        trace = generate_trace(scenario)
        for ewma, cooldown, residence in ((30, 60, 90), (60, 60, 90), (300, 60, 90),
                                          (30, 120, 90), (30, 60, 180)):
            profile = replace(SHORT_EXPERIMENT, ewma_s=ewma, cooldown_s=cooldown, residence_s=residence)
            result = run(scenario, configs, profile=profile, records=trace)
            output.append({"scenario": name, "ewma_s": ewma, "cooldown_s": cooldown,
                           "residence_s": residence, "goodput": result["simulated_slo_goodput"],
                           "unmet": result["simulated_unmet_demand"], "transitions": result["transitions"]})
    return {"schema_version": SCHEMA_VERSION, "kind": "simulation-sensitivity", "results": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=SCENARIOS + ("all",), default="all")
    parser.add_argument("--profile", choices=("short-experiment", "conservative"), default="short-experiment")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = CONSERVATIVE if args.profile == "conservative" else SHORT_EXPERIMENT
    configs = configurations()
    names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    result = {"schema_version": SCHEMA_VERSION, "kind": "simulation-suite", "profile": asdict(profile),
              "comparisons": [compare(load_scenario(ROOT / "configs" / "workloads" / f"{name}.json"), configs, profile)
                              for name in names]}
    if args.sensitivity:
        result["sensitivity"] = sensitivity(configs)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
