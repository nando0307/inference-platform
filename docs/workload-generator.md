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

Relative monotonic offsets drive scheduling. Wall-clock timestamps are not used for latency calculations. Results distinguish scheduled, submitted, completed, rejected, canceled, timed-out, transport-failed, and client-limit requests. Streaming timestamps identify observed SSE events and first text, not exact engine token generation time. A streamed response succeeds only after `[DONE]`; malformed JSON, a model identity mismatch, an event after `[DONE]`, or an incomplete stream is recorded as an error. Requests still running when the drain timeout expires are canceled and included in the results.

Replay results retain the trace `request_id` for comparison and store any server `x-request-id` separately as `server_request_id`. They carry the trace's declared `model_revision`; this does not verify the revision actually loaded at the endpoint. Older result files that replaced trace IDs with server IDs cannot be matched; replay the trace again.

## Compare two replays

Replay the **same** trace against both endpoints (with the same replay settings), then run:

```bash
.venv/bin/python -m benchmarks.workload_generator.compare \
  --trace traces/mixed-15min-seed42.jsonl \
  --baseline results/placement-a.jsonl \
  --candidate results/placement-b.jsonl \
  --output results/a-vs-b.json
```

The comparison preserves both raw results per trace request, including errors, 429 rejections, cancellations, timeouts, client-limit omissions, and absent results. Its `groups` include the overall run and each phase/model; each side gives scheduled/result/status counts and measured-sample distributions (`n`, p50, p95, min, max) for completion latency, streaming first text, submission delay, and successful token counts. `candidate_minus_baseline` subtracts corresponding p50/p95 values only when both exist; these are descriptive, not confidence intervals or paired causal estimates. Missing eligible metrics and results generate warnings, never zero values. Conflicting declared model revisions or trace identities are rejected; missing older result revisions are warned about. Actual loaded server revision cannot be confirmed from replay artifacts alone. Keep workload timing and hardware comparable before interpreting performance differences; accelerated `--time-scale` runs are not real-time performance comparisons. The suite does not calculate confidence intervals, goodput, cost, or hardware performance conclusions.
