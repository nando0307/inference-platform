# Verified results summary

Report and CV source of record. Every number below was read from a saved
artifact or an executed check; none are estimates. Artifacts live under
`benchmark-results/` and are intentionally gitignored. Source state at the time
of writing: commit `89e2755`, 182 passing tests.

Do not reuse these numbers without their scope labels. Several are single
observations, cross-platform ratios, or include model loading and queueing.

## Environments

| Name | Details |
|---|---|
| Local CPU | Apple M1 Pro, 8 logical CPUs, 16 GB RAM, macOS 26.6.2 arm64, Python 3.12.13, torch 2.14.1, transformers 5.18.0, FastAPI 0.143.3, uvicorn 0.54.0, httpx 0.28.1. A clean venv from `requirements.txt` resolved the same direct pins with tokenizers 0.23.3, numpy 2.5.4, and huggingface-hub 2.4.0 |
| Cloud GPU | NVIDIA Tesla T4 15360 MiB, 4 vCPU and 8192 MiB host memory per container, Linux gVisor x86_64 container, vLLM 0.33.0 pinned image digest, torch 2.14.0+cu130, transformers 5.20.2 |
| Shared model | `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| Second model | `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |

## Headline numbers

| Area | Metric | Value | Scope | Artifact |
|---|---|---|---:|---|
| Test suite | Automated tests | **182 passed, 0 failed** | Local, ~6 s, fake inference double plus loopback ASGI | `tests/` |
| Fresh environment | Clean venv install and run | install **27.5 s**; **182 tests passed**; `/ready` 200; one completion 200 | Python 3.12.13, exact pins from `requirements.txt`, cached model | `logs/practice.md#fresh-environment-acceptance` (recorded walkthrough observation, 2026-11-08) |
| Static batching CPU | 4 prompts, 44 completion tokens total per workload, median wall time | FP32 batch 1: **4483.37 ms**; batch 4: **2193.14 ms**; **2.04x** faster | Model-only workload wall time, one warm-up excluded, 2 measured samples per configuration | `benchmark-results/batching/cpu-benchmark-001.json` |
| Static batching T4 | Same 4-prompt, 44-token workload, median wall time | FP32 batch 1: **1062.62 ms**; batch 4: **465.15 ms**; **2.28x** faster | CUDA-synchronized model-only workload wall time, one warm-up excluded, 3 measured samples per configuration | `benchmark-results/batching/t4-benchmark-001.json` |
| Static batching T4 | Aggregate output throughput | **41.17 tok/s** at batch 1; **94.26 tok/s** at batch 4 | Successful completion tokens / successful measured workload seconds, summed across 3 measured samples per configuration; 44 tokens per sample | `benchmark-results/batching/t4-benchmark-001.json` |
| Static batching T4 | Peak allocated memory | **1.99 GB** batch 1; **2.01 GB** batch 4 | PyTorch CUDA allocator, weights included, peak over 3 measured samples per configuration | `benchmark-results/batching/t4-benchmark-001.json` |
| Phase 4 concurrency | Short prompt, short output, p50 E2E | c=1: **2459.2 ms**; c=2: **3731.7 ms**; c=4: **4881.0 ms** | HTTP request duration, one worker, two waiting slots | `benchmark-results/phase4/report-c1.json`, `report-c2.json`, `report-c4.json` |
| Phase 4 concurrency | Useful output throughput, short/short | **12.98 / 12.92 / 13.05 tok/s** at c=1/2/4 | Successful output tokens per scenario second | `benchmark-results/phase4/report-c1.json`, `report-c2.json`, `report-c4.json` |
| Phase 4 concurrency | Queue overflow at c=4 | **5 of 20 requests rejected** with 429 in each case | Two waiting slots; running request not counted | `benchmark-results/phase4/report-c4.json` |
| KV cache | Scaling cases matched | **5 of 5** cases matched predicted logical bytes and same-device HF token IDs | 24 layers, 2 KV heads, 64 head dim; CPU FP32 and T4 FP16 | `benchmark-results/cache-scaling/` |
| Precision CPU | Decode speed, short prompt | FP32 **88.18 ms/token** (11.34 tok/s); FP16 **30.47 ms/token** (32.82 tok/s); **2.89x** FP16 speedup | One warm workload, greedy, CPU M1 Pro | `benchmark-results/precision/cpu-precision-001.json` |
| Precision CPU | Decode speed, long prompt | FP32 **108.73 ms/token**; FP16 **53.87 ms/token**; **2.02x** speedup | Same protocol | `benchmark-results/precision/cpu-precision-001.json` |
| Precision CPU | Dynamic INT8 decode | Short **48.33 ms/token**; long **60.88 ms/token** | Slower than FP16 on both workloads | `benchmark-results/precision/cpu-precision-001.json` |
| Precision CPU | Frozen quality tasks | FP32 **8/9**; FP16 **8/9**; dynamic INT8 **1/9** | 3 arithmetic, 3 facts, 3 instructions; greedy | `benchmark-results/precision/cpu-precision-001.json` |
| Warm single-stream T4 | Median inference duration | **3397.13 ms**; **36.86 output tok/s** | 59 prompt / 128 max tokens, 5 warm measurements | `benchmark-results/modal/warm-t4-001.json` |
| Matched CPU control | Median inference duration | **11604.62 ms**; **10.96 output tok/s** | Same workload and client on M1 Pro | `benchmark-results/modal/cpu-matched-001.json` |
| Cross-platform ratio | Median duration ratio | **3.42x** T4 vs M1 Pro (3.36x on throughput) | Different hardware and stacks; not an isolated GPU speedup | `benchmark-results/modal/warm-t4-001.json`, `benchmark-results/modal/cpu-matched-001.json` |
| Engine comparison | Sequential output throughput | Transformers **43.19 tok/s**; vLLM **42.47 tok/s**; vLLM **-1.67%** | Same T4, FP16, Qwen2.5-0.5B, 176 tokens per scenario | `benchmark-results/engine-comparison/t4-transformers-vllm-001.json` |
| Engine comparison | Refill output throughput | Transformers **43.63 tok/s**; vLLM **58.98 tok/s**; vLLM **+35.19%** | One matched T4 scenario; C admitted when A completes; fixed order | `benchmark-results/engine-comparison/t4-transformers-vllm-001.json` |
| Engine comparison | C first text under refill | Transformers **2987.20 ms**; vLLM **46.77 ms** | From C's own request start | `benchmark-results/engine-comparison/t4-transformers-vllm-001.json` |
| Refill, vLLM only | Sequential vs refill throughput | **42.18 vs 38.89 tok/s**; refill **-7.81%** | Older single-engine run; warm-up and order differ | `benchmark-results/continuous-batching/t4-vllm-001.json` |
| Refill, vLLM only | B still active when C finishes | C done at **2625.34 ms**, then B done at **4525.47 ms** on scenario clock; C request E2E **772.90 ms** from C's own start | Visible refill, not a throughput win in this sample | `benchmark-results/continuous-batching/t4-vllm-001.json`; timeline detail: `logs/practice.md#measured-results-and-refill-timeline` |
| Two-model routing | Model startup times | **73560.54 ms** (Qwen2.5-0.5B), **65881.53 ms** (Qwen3-1.7B), **48008.74 ms** (reload Qwen2.5) | T4, pinned image, cold process group per load | `benchmark-results/model-routing/t4-two-model-lifecycle-001.json` |
| Two-model routing | Process-group unload | **605.72 / 505.42 / 556.30 ms**, all unforced, exit 0 | SIGTERM to complete group exit | `benchmark-results/model-routing/t4-two-model-lifecycle-001.json` |
| Two-model routing | Whole-device memory | **6 MiB idle**, **7612 MiB** low resident, **7608 MiB** high resident, **6 MiB** after cleanup | `nvidia-smi`, includes KV reservation at `--gpu-memory-utilization 0.5` | `benchmark-results/model-routing/t4-two-model-lifecycle-001.json` |
| Two-model routing | Frozen quality tasks | Qwen2.5-0.5B **8/9**; Qwen3-1.7B **9/9** | 18 greedy samples, 9 tasks per model | `benchmark-results/model-routing/t4-two-model-lifecycle-001.json` |
| Two-model routing | Warm short-prompt first text | Low model about **31.5 ms**; high model **37.9-47.8 ms** | After load, short outputs | `benchmark-results/model-routing/t4-two-model-lifecycle-001.json` |
| Two-model routing | Requests issued during a switch | **48.6-74.8 s** end to end | Includes drain, model load, and admission | `benchmark-results/model-routing/t4-two-model-lifecycle-001.json` |
| Two-model routing | Attempts and failures | **26 attempts**: 25 completed, 1 queued cancellation, 0 failures | 3 loads, 32-attempt budget | `benchmark-results/model-routing/t4-two-model-lifecycle-001.json` |
| Operations | Completion latency and throughput | **1104.06 ms** request; **3.629 completion tok/s**; 36 prompt / 4 completion tokens | Local CPU clean venv, one non-stream request | `logs/practice.md#fresh-environment-acceptance` (recorded walkthrough observation, 2026-11-08) |
| Operations | Process peak RSS | **2.32 GB** after one completion | Clean-venv high-water RSS, includes model | `logs/practice.md#fresh-environment-acceptance` (recorded walkthrough observation, 2026-11-08) |
| Operations | Guards | 64 KiB body cap, 401 without bearer token, 503 metrics before startup, counters reset on restart | Local service | `logs/practice.md` |

## CV-ready bullets

Use the scope notes above; these are not general speedup claims.

- Built a FastAPI inference service with SSE streaming, bounded concurrency
  (2 waiting slots), cancellation, request isolation, and 182 automated tests.
- Reduced model-only wall time for four prompts generating 44 completion tokens
  total per measured workload by **2.28x** on a T4 (FP32, 1062.62 to
  465.15 ms) and **2.04x** on an M1 Pro (FP32, 4483.37 to 2193.14 ms) by
  comparing batch sizes 1 and 4.
- In one matched Transformers-versus-vLLM scenario on one T4, vLLM reached
  **58.98 output tok/s** versus **43.63** under the refill policy (+35.19%),
  while sequential throughput was effectively flat (-1.67%).
- Cut CPU FP16 decode time **2.89x** versus FP32 for a short prompt (88.18 to
  30.47 ms/token) with unchanged 8/9 frozen quality results; dynamic INT8 was
  slower and dropped to 1/9.
- Implemented single-resident routing for two pinned checkpoints with a bounded
  FIFO queue; measured **48-75 s** model-switch costs and **9/9** versus **8/9**
  frozen quality tasks for Qwen3-1.7B versus Qwen2.5-0.5B.
- Added a metrics snapshot (`requests`, `errors`, latency percentiles, queue
  depth, token throughput, RSS), optional bearer auth, and a 64 KiB body cap;
  verified restart, failed startup, 401, and 413 behavior on the real model.

## Claims to avoid

- No isolated GPU or engine-speedup claim. T4-versus-M1 Pro is cross-platform;
  engine comparison is a whole-stack, single observation with fixed order.
- No throughput claim from the older vLLM-only refill run; it regressed by 7.81%.
- No KV-cache memory-saving claim; measurements validated logical cache bytes and
  token IDs, not allocator savings.
- No quantization speedup claim for INT8; it was slower and lost quality.
- No model-quality ranking from 9 frozen greedy tasks, and no latency ranking
  from single observations.
- No concurrency speedup claim on CPU: useful throughput stayed near 13 tok/s
  while latency and rejections grew.

## Regenerate

```bash
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m benchmarks.client --help
```

Each artifact retains its own runtime, request rows, checks, and limits. Reuse
the paths above rather than rerunning paid GPU jobs; any rerun needs a fresh
artifact path and approval.
