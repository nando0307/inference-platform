# Rust control plane (schema v1)

Independent library for configuration, typed worker identity/registry, deterministic routing, atomic admission and reservation ownership. It has no Python binding or production integration protocol: Python still owns the gateway, worker execution/lifecycle/health, stream cleanup, and inference. It does not depend on the unfinished placement executor and starts no processes or GPUs. Capacity weights accept positive arbitrary-precision integers, matching Python's schema-v1 behavior. `unicode-ident` supplies the XID character properties used by Python's `str.isidentifier()`; `unicode-normalization` enforces Python's NFKC hostname delimiter check. Rust's standard library has neither.

`ServingConfig::load` validates the existing A–D JSON topology before constructing identities. `Admission::reserve` holds one mutex across eligibility, policy selection, slot assignment, credit/cursor commit and counters; it returns a non-cloneable `Reservation`. Call `dispatchable` just before handing work off, then `release` once. Dropping the owner (including an aborted future) cancels it. `advance_generation` retires active/queued reservations as failed and fences old owners and stale health updates; a failed worker cannot become ready in the same generation. External/async work must occur **outside** the mutex. No lifecycle work belongs in this crate.

## Checks

```sh
cargo test --manifest-path rust-control-plane/Cargo.toml
PYTHONPATH="$PWD" python3 rust-control-plane/parity.py
python -m unittest tests.test_worker_manager tests.test_serving_config tests.test_routing -q
```

Parity compares deterministic Python-versus-Rust fixtures: valid A–D config and strict invalid mutations, edge numeric/URL cases, all four selector policies/ties/credits, and B/D manager traces including queue saturation, FIFO releases, cancellation and per-worker counters. Rust tests cover atomic concurrent reservations, unique owners, generation races, FIFO, drop/abort accounting and state fencing. Checks do not start workers.

## CPU-only benchmark (2026-11-27)

Run after parity, with the Python dependencies installed:

```sh
python rust-control-plane/run_benchmarks.py /tmp/rust-control-bench
# Raw per-repeat nanosecond samples: /tmp/rust-control-bench/raw.jsonl
# Hardware, toolchains, aggregate data: /tmp/rust-control-bench/summary.json
```

Committed measurements: [`benchmarks/raw.jsonl`](benchmarks/raw.jsonl) and [`benchmarks/summary.json`](benchmarks/summary.json). Apple M1 Pro (8 logical CPUs, 16 GiB), macOS 26.6.2 arm64, Python 3.12.13, rustc/cargo 1.95.0. Rust `--release --locked`; same B config, small model, least-inflight, two eligible replicas, no GPU/network. Each workload runs one warmup then five measured repeats in a separate process per language. Selector: 1,000 warmup/10,000 measured per repeat, fixed two-candidate snapshot. Reservation: 300/3,000, submit/reserve only timed, cleanup outside sample. Synthetic: 50/500 cycles, each attempts nine small-model requests on eight bounded slots (2 active + 2 FIFO per replica), checks one `server_busy`, performs mock preparation/dispatch, releases eight; synthetic sample is one nine-request cycle. Python uses its actual `MultiWorkerManager` with an in-memory no-op engine and asyncio; Rust uses `Admission` with no async task/runtime. Peak RSS is process-level `/usr/bin/time -l` **maximum resident set size** (bytes), so Python imports and its asyncio runtime are included; it is not incremental library memory. P50/P95/P99 use nearest rank over all raw per-operation samples; throughput uses total wall elapsed across measured repeats including cleanup (nine attempted requests per synthetic cycle). No warming or measurement starts worker processes.

| Workload | Language | P50/P95/P99 (µs) | Attempt throughput (/s) | Peak RSS (MiB) | Capacity / rejected |
|---|---|---:|---:|---:|---:|
| selector | Python | 0.583 / 0.625 / 0.708 | 1,510,924 | 336.3 | — |
| selector | Rust | 0.042 / 0.084 / 0.125 | 11,190,171 | 2.5 | — |
| reservation | Python | 4.208 / 4.916 / 41.208 | 82,667 | 338.3 | — |
| reservation | Rust | 0.333 / 0.375 / 0.458 | 2,303,410 | 2.3 | — |
| synthetic | Python | 116.916 / 180.500 / 238.916 | 71,274 | 336.8 | 8 / 2,500 |
| synthetic | Rust | 3.708 / 4.292 / 14.000 | 2,239,410 | 2.2 | 8 / 2,500 |

These are CPU microbenchmarks, **not** equivalent gateway/worker/inference implementations or production-capacity/speedup claims. Python submission creates/cancels asyncio preparation tasks and imports production dependencies; Rust deliberately does neither. Synthetic work substitutes a no-op engine and excludes HTTP, model execution, stream cleanup and GPU time. Real integration would require a separately reviewed bridge and new end-to-end measurements.
