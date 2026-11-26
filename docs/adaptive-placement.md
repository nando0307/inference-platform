# Adaptive placement: simulation and dry-run only

`src/placement_planner.py` produces versioned recommendations. `benchmarks/placement_simulation.py` is a separate CPU-only discrete-event experiment. Neither module calls a serving endpoint, allocates a GPU, starts/stops/replaces a worker, changes routing, or executes a recommended placement. Never interpret these outputs as a migration plan safe to execute on hardware.

## Repository inputs and provenance

`configs/serving-{a,b,c,d}.json` are loaded through `src.serving_config.load_config` (including pinned revisions, physical GPU UUIDs and CUDA ordinals). Their actual worker mappings are: A = `small-0` on GPU 0, `large-1` on GPU 1; B = `small-0` + `large-0` on GPU 0, `small-1` on GPU 1; C = `small-0` + `large-0` on GPU 0, `large-1` on GPU 1; D = `small-0` + `large-0` on GPU 0. These are configuration declarations, **not measured capacity**. The same two physical UUIDs form the fixed budget for every policy. Configured `memory_utilization: 0.5` is a per-process budget, not a measured model footprint. No GPU memory telemetry, worker health, or observed SLO capacity is inferred from that field. The optional serving `/metrics` exposes worker readiness and GPU memory, but this experiment does not consume live metrics.

The simulator separately assumes a per-model demand bound of **100 requests/s**, dedicated replica service rates of **0.7 small / 0.45 large requests/s**, a **0.7 multiplier per worker when sharing one GPU**, footprints **0.35 small / 0.40 large of GPU memory**, a **10-second completion SLO**, and **two simultaneous + two queued requests per worker**. All are illustrative; co-residency, throughput, latencies, and footprints have **not** been hardware-validated. Per-worker service seconds/request is `1 / (dedicated_rps × shared_multiplier_if_colocated)`. Every policy uses identical rates, footprints, scenario traces, SLO, concurrency, and service rule. Prompt/output lengths and cancellations come from the existing seeded workload trace; token lengths are retained in trace metadata but this simplified service rule does not predict token-specific latency. No single-T4 smoke result establishes capacity here.

## Schemas and pure planner

`Snapshot`, `Profile`, `History`, `Step` and `Recommendation` are frozen Python dataclasses. Snapshot and recommendation have `schema_version: 1`; `Recommendation.to_dict()` is JSON serializable. Snapshot identifies observation boundary `at_s`, separate sample timestamps and provenance (`assumed`, `simulated`, or `measured`) for incoming per-model RPS (with an explicit per-model upper bound), per-replica SLO-compliant RPS, and per-replica GPU memory fractions; `worker_ready` distinguishes true, false, unknown. `shared_gpu_factor` is explicitly supplied. Revision and GPU UUID consistency are checked against loaded configurations. Unknown or missing required fields and samples older than `max_age_s` **hold**, never silently become zero. An explicit observed zero is valid. Demand above its bound, future timestamps, time reversal, invalid numeric values and unsupported provenance/schema are rejected. Passing a `measured` label is a caller assertion, not measurement by the planner.

For each candidate, steady-state feasibility requires at least one configured and ready replica of every required pinned model; for each physical GPU, sum of replica memory fractions must not exceed `1 - memory_reserve`. Separately, the transition search starts with current workers, adds workers before removing workers when safe, checks the full intermediate resource budget and required-model availability after every step, and rejects paths with no safe next move. An addition is not available until its step completes; a removal remains available until its step completes. Existing simulated requests drain before removal. The actual service never follows this plan. Unknown readiness of even an inactive candidate worker holds all recommendations rather than speculating it can start.

At horizon `H` seconds, placement score (requests) is `H × Σ_model min(estimated_demand_rps, placement_capacity_rps)`. Each transition step of assumed `impact_s` seconds incurs `impact_s × Σ_model max(0, min(demand, candidate_capacity) - min(demand, step_capacity))` **estimated lost candidate requests**. The candidate net score subtracts these losses, and must strictly exceed the current score by at least 15% (strictly exceed the threshold, not equal it). The total assumed transition duration must fit inside the horizon. Losses cannot be offset by transient capacity gains. The step estimate uses *pre-step* serving capacity for the entire impact interval, matching the simulator's membership change at step completion (and drain wait before removal); estimated loss is still not an observed loss. This is a dimensional conversion from seconds × requests/second to requests, not a hardware measurement. The current placement wins ties; among equally scoring eligible candidates, placement name is ascending. All candidates have explicit rejection reasons and scores when steady-state feasible. A recommendation includes its transition steps, costs, assumptions, provenance, timestamps, current/candidate placement and updated immutable history.

## Observation and time semantics

Time is monotonic simulated seconds from scenario start. Observation window `i` is `[(i-1) × interval_s, i × interval_s)`; arrivals at its right boundary belong to the *next* window. At boundary `i × interval_s`, queued completions and transition steps are resolved first, then the previous interval's arrival counts produce one demand sample, then arrivals at the boundary are processed. No service outcomes, phase definitions, future scheduled requests, or later telemetry enter a recommendation. A missing/stale window clears the demand filter and candidate streak; next valid sample initializes EWMA to its observed RPS, not zero. On contiguous valid windows, `alpha = 1 - exp(-interval_s / ewma_s)` and `next = previous + alpha × (observed - previous)`. A missing interval also resets the streak. Three consecutive eligible windows must agree on the same candidate; otherwise the candidate/streak restarts. For standalone dry-run recommendations, `last_change_s` is the recommendation timestamp until the caller supplies an observed placement activation time; in the simulator it is replaced with the completion time of the final transition step. Cooldown and minimum residence start at that activation time; another switch is forbidden while `t - last_change_s < max(cooldown_s, residence_s)`; equality allows it. The initial placement has no historical switch timestamp and is eligible after the observation gate. A simulated transition remains pending until all its steps finish, and further planner calls pause during it. Delayed drains may make actual simulated impact exceed the fixed planning estimate.

Profiles are **illustrative simulation settings, not production settings**:

| Setting | conservative | short-experiment |
|---|---:|---:|
| EWMA timescale | 300 s | 30 s |
| Evaluation interval | 60 s | 10 s |
| Consecutive windows | 3 | 3 |
| Improvement threshold | 15% | 15% |
| Cooldown | 600 s | 60 s |
| Minimum residence | 900 s | 90 s |
| Memory reserve | 20% | 20% |
| Planning horizon | 300 s | 300 s |
| Assumed transition step | 10 s | 10 s |
| Maximum telemetry age | 120 s | 20 s |

Each `Profile` field is configurable with `dataclasses.replace`; all defaults are explicit in machine-readable outputs. Horizon, step impact, freshness bounds and evaluation interval are simulation choices, not production recommendations.

## Run and inspect

```bash
python -m unittest tests.test_placement_planner -v
python -m benchmarks.placement_simulation --scenario all --profile short-experiment \
  --sensitivity --output /tmp/placement-simulation.json
python -m benchmarks.placement_simulation --scenario mixed_15min --profile conservative \
  --output /tmp/placement-conservative.json
```

All outputs are schema-versioned JSON and explicitly labeled `simulation-suite`, `simulation-comparison`, `simulation`, or `simulation-sensitivity`. The suite compares static A/B/C/D and adaptive starting from A on the *same generated trace*, with no live deployment or warmup (`warmup_s: 0`, measurement over the scenario duration). It reports offered requests, simulated SLO goodput (completed within 10 s, not canceled), unmet demand, rejected, late/canceled, switch count, and separately **estimated** transition-lost requests. Work already admitted can complete after the measurement end; offered arrivals are limited to the declared duration. Scenarios are `balanced`, `small_peak`, `large_peak`, `alternating_peaks`, `sudden_spike`, `microburst`, `overload`, `idle`, and `mixed_15min`; missing/stale telemetry can be exercised end-to-end in the static-versus-adaptive comparison with `--telemetry-gap missing` or `--telemetry-gap stale`; both run the identical trace, while unavailable adaptive inputs produce explicit hold recommendations and preserve placement. The Python `run(..., missing_windows=(...), stale_windows=(...))` interface supports targeted injection. The sensitivity run varies EWMA (30/60/300 s), cooldown (60/120 s), and residence (90/180 s) across small peak, large peak, alternating peaks and mixed 15-minute scenarios. Its comparisons are exploratory, not parameter optimization or causal hardware evidence.

Limits: no real service-time distribution, token-length dependence, GPU scheduler contention measurement, observed demand integration, noisy telemetry calibration, request retry behavior, queuing calibration, executor, scaling, RL, tensor parallelism, GPU kernels, or performance/SLO claim. The planner cannot prove that a theoretical transition is executable safely in the live worker lifecycle; the only supported action is inspecting a dry-run recommendation.
