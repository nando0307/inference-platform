# Reproducible workload generator

The workload generator is separate from benchmark analysis. It creates a deterministic JSONL request trace, then replays that same trace against any public inference endpoint, including static placements A-D.

## Generate a trace

```bash
.venv/bin/python -m benchmarks.workload_generator validate \
  --scenario configs/workloads/mixed_15min.json

.venv/bin/python -m benchmarks.workload_generator generate \
  --scenario configs/workloads/mixed_15min.json \
  --output traces/mixed-15min-seed42.jsonl
```

Generation writes two files:

- `*.jsonl`: one self-contained request per line.
- `*.meta.json`: schema, model revisions, seed, corpus ID, and request count.

The sidecar is required for replay and is checked before requests are sent.

## Replay

```bash
INFERENCE_API_TOKEN=... \
.venv/bin/python -m benchmarks.workload_generator replay \
  --trace traces/mixed-15min-seed42.jsonl \
  --endpoint http://127.0.0.1:8000 \
  --output results/placement-a.jsonl \
  --max-inflight 64 \
  --max-connections 64
```

Replay is open-loop. Completion does not alter future scheduled arrivals. If the client reaches `max-inflight`, the request is recorded as `not_submitted` rather than silently converted into a closed-loop wait. `--time-scale` is for local development only; accelerated runs are not comparable with real-time performance measurements.

The generator uses independent deterministic random streams for arrival timing, model selection, prompt selection, output length, streaming selection, and cancellation. Changing prompt generation does not change the arrival schedule or model choices.

## Scenarios

The repository includes balanced, model-specific peak, spike, ramp, alternating peak, microburst, overload, idle, token-length, cancellation, 15-minute, and 60-minute scenario definitions under `configs/workloads/`.

The primary `mixed_15min` scenario contains:

- 0-180 seconds: balanced traffic.
- 180-360 seconds: small-model peak.
- 360-540 seconds: large-model peak.
- 540-720 seconds: heavy mixed traffic.
- 720-900 seconds: zero incoming traffic for drain observation.

Rates are synthetic offered-load defaults. They are not T4 capacity measurements.

## Trace and result semantics

Relative monotonic offsets drive scheduling. Wall-clock timestamps are not used for latency calculations. Results distinguish scheduled, submitted, completed, rejected, canceled, timed-out, transport-failed, and client-limit requests. Streaming timestamps identify observed SSE events and first text, not exact engine token generation time.

The generator does not calculate placement comparisons, confidence intervals, goodput, cost, dashboards, or performance conclusions. Those belong to `feat/benchmark-suite`.
