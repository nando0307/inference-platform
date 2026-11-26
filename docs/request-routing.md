# Static worker request routing

The static multi-worker gateway reads `routing.policy` from the existing schema-v1 serving config.
Omit it to keep the previous default, `least-inflight`.
Valid values are `least-inflight`, `least-queue-depth`, `round-robin`, and `weighted-capacity`.
Each worker optionally has a positive integer `capacity_weight` (default `1`); only weighted-capacity uses it.
For example, set `"policy": "weighted-capacity"` in `routing` and `"capacity_weight": 3` on a worker with an explicitly assumed threefold service capacity.
Weights are operator assumptions, not inferred from GPU memory, T4 smoke results, or measured throughput.

Only ready workers serving the exact requested model and having an active or bounded queue slot are candidates.
The manager snapshots candidates, selects synchronously, and reserves the lease before yielding to the event loop.
Lifecycle and generation checks remain in preparation and forwarding; an unavailable worker is never replayed to another worker.
A model with ready but saturated replicas returns 429 `server_busy`; one without a ready replica returns 503 `not_ready`.
Queues, cancellation, timeout, shutdown, and backpressure retain their existing behavior.

| Policy | Choice from eligible workers | Trade-off |
|---|---|---|
| least-inflight | Fewest active plus queued leases, then worker ID | Existing default; ignores varying service speeds. |
| least-queue-depth | Fewest queued leases, then worker ID | Prefers a worker with an empty queue even if its active slots are busy. |
| round-robin | Next worker ID after the last successfully reserved worker, wrapping around | Ignores load; cursor is scoped per model and does not move on rejection. |
| weighted-capacity | Smooth weighted round-robin with per-model credits over eligible workers, then worker ID | Proportional to configured weights when all replicas are eligible; ineligible workers do not accrue credit. |

The authenticated `/metrics` response adds a bounded `routing` group with `policy`, cumulative `outcomes` (`reserved`, `server_busy`, `not_ready`), and a 256-sample `decision_ms` window for successful selection and reservation.
These process-local counters count reservation decisions, not completed HTTP responses; the existing `http` metrics remain authoritative for delivery outcomes.
No request-ID or arbitrary user-supplied metric labels are introduced.

## Deterministic simulation, not hardware evidence

Run `python -m benchmarks.routing --seed 17` from the repository root with the existing CPU dependencies.
The script reuses `benchmarks.workload_generator.arrivals`, the schema-v1 B placement, and the real `MultiWorkerManager` admission logic with simulated workers and virtual service time.
All four policies see the same seeded arrivals and model demands in each scenario.
The assumptions are 6 virtual seconds, Poisson arrivals at 6 requests/s (20 for saturation), a one-second simulated lease duration, two active and two waiting slots per worker, and weight 3 with one-third-second virtual service on `small-0` for the uneven-capacity scenario.
Concurrent arrivals are bucketed at 0.1-second boundaries; changing-demand switches requested models halfway through; failure-draining fails `small-0` after 2 seconds and drains `small-1` after 4 seconds.
Output records `evidence: simulation`, scenario, seed, policy, assumptions, admission outcomes, and assignments.
The script does not issue HTTP requests or execute GPU inference; its `simulated_released` counter represents virtual lease removal, not a completed generation.

For seed 17, balanced traffic assigns small requests to `small-0`/`small-1` as 10/8 under least-inflight, 9/9 under round-robin and equal-weight routing, and 14/4 under least-queue-depth.
In the assumed uneven-capacity case, the counts are 14/4 for least-inflight and weighted-capacity, 9/9 for round-robin, and 18/0 for least-queue-depth.
Saturation produces 38 simulated 429 responses out of 104 arrivals under each policy; changing model demand produces 4 simulated 429 responses out of 27 arrivals.
These are discrete-event **simulation results**, not GPU or HTTP latency measurements; equal aggregate admissions here do not establish sustained throughput, production capacity, latency SLO compliance, or two-GPU validation.
The existing single-T4 smoke record supplies no capacity weight or validation of this simulation's service-time assumption.
