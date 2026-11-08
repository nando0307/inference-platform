```text
```
# inference platform
Hi i am Do, i am learning about the inference !
my dream is to becoming a llm engineer xD
```
```

## Current learning checkpoint

Read [the session handoff in logs/practice.md](logs/practice.md) before continuing
with a new mentor. It records inference experiments, verified API checkpoints,
and the next task. Use [plan.md](plan.md) for the broader roadmap.

The implementation now includes Phase 4 controlled concurrency: one inference
worker, two FIFO waiting slots, cooperative cancellation, and bounded streaming.
The suite has 182 passing tests, with real-Qwen lifecycle and HTTP smoke evidence.
See the [Phase 4 evidence](logs/practice.md#phase-4-bounded-concurrency-and-cancellation)
for the fixed-workload comparison and its limitations. Stage 5 now includes CPU
and T4 static-batching measurements at sizes 1/2/4; the API remains sequential.
Stage 6 has completed [CPU and T4 dynamic-cache scaling experiments](#dynamic-kv-cache-scaling).
All five T4 cases matched the CPU logical-cache bytes and passed same-device
Hugging Face token-ID comparisons. A [CPU dynamic/static allocation comparison](#dynamic-versus-static-kv-cache-allocation)
also passed with identical outputs and distinct used/capacity measurements.
Independent learning checks remain pending; automated verification does not
prove understanding.
Stage 7 now includes a [CPU FP32/FP16/dynamic-INT8 comparison](#cpu-precision-and-quantization):
six fixed-length speed observations and nine frozen quality tasks per format.
FP32 and FP16 passed 8/9 tasks; QNNPACK dynamic INT8 passed 1/9, with instruction-following regressions.
A [real T4 vLLM experiment](#continuous-batching-with-vllm) now demonstrates
request refill while another stream continues. The existing API remains sequential;
the engine comparison runs separately and does not establish a throughput improvement.
A [matched Transformers-versus-vLLM runner](#transformers-versus-vllm-comparison)
now reuses that client against both servers without changing the production API.
The earlier GPU exploration completed a five-sample warm T4 run:
3397.13 ms median inference duration and 36.86 output tokens/s for the 59/128 case.
The matched-library CPU control observed 11604.62 ms and 10.96 output tokens/s.
The resulting 3.42x median ratio is cross-platform evidence, not an isolated
hardware-speedup claim.
Stage 9 now adds an opt-in [explicit two-model router](#explicit-two-model-routing)
with one resident model, a bounded FIFO queue, pinned tokenization, and validated
JSON/SSE responses. Two approved T4 runs exercised a Qwen3 smoke and the full
two-model lifecycle: three loads, cross-model FIFO, queued cancellation, a clean
process-group shutdown, and frozen quality tasks (8/9 low, 9/9 high).
Stage 10 now adds a [metrics snapshot](#metrics) endpoint, optional bearer
authentication, a 64 KiB request-body cap, and a verified local restart and
failed-startup procedure. The roadmap in [plan.md](plan.md) ends here; the
remaining items are independent learning checks. Report-ready numbers and their
scopes live in [the results summary](logs/results-summary.md).

## Explore the codebase with Understand Anything

[Understand Anything](https://github.com/Egonex-AI/Understand-Anything) is a
separate developer tool, not part of the inference API or Python dependencies.
The project graph lives in `.ua/knowledge-graph.json`, with architectural layers
and a guided tour covering source, tests, and learning documentation.
`.ua/.understandignore` excludes secrets, environments, model weights, and scratch files.

### In Pi

The official installer downloads the tool to `~/.understand-anything/repo` and
links its skills into `~/.agents/skills` (available across projects). On a new
machine, install Node.js 22+ and pnpm 10+, then:

```bash
git clone https://github.com/Egonex-AI/Understand-Anything.git ~/.understand-anything/repo
bash ~/.understand-anything/repo/install.sh pi
```

Restart Pi in this repository, then use:

```text
/skill:understand
/skill:understand-dashboard
/skill:understand-chat Explain the streaming request lifecycle
```

Analysis uses your coding agent/model; it does not call this project's Qwen API.
Re-run `/skill:understand` after code changes; use `/skill:understand --full` to
force a rebuild, including uncommitted edits at the same Git commit. Automatic
updates are disabled; no Git hooks are installed.

### Dashboard without an agent

The v2.9.7 release viewer download was unavailable, so this installation uses
the locally built viewer. On a new machine, build it once after installation:

```bash
cd ~/.understand-anything/repo
npx --yes pnpm@10.6.2 install --frozen-lockfile
npx --yes pnpm@10.6.2 --filter understand-anything-viewer build
```

Then, from this project's repository root:

```bash
node "$HOME/.understand-anything-plugin/packages/viewer/bin/viewer.mjs" "$PWD"
```

Open the full localhost URL printed by the viewer, including `?token=...`.
The viewer is local and read-only; no LLM/API key is needed. Stop it with Ctrl-C.
Commit the graph, config, metadata, fingerprints, and ignore rules in `.ua/` to
share the analysis; scratch output is ignored by Git. Review generated summaries
before sharing them outside the project.

## Bounded warm Qwen measurements on Modal

`modal_inference.py` is a GPU exploration checkpoint after Phase 4, not an API
deployment. It loads the pinned Qwen runtime once onto a T4 in float32, then
reuses `src.inference.generate()` sequentially for two warm-ups and five measured
generations. Model weights are shared; generation state and KV caches are not.
The local CPU service and CPU-only token instrumentation remain unchanged.

The experiment uses `short-input-long-output` from the existing
`benchmark-results/phase3/workload.json`: 59 formatted prompt tokens, a 128-token
output cap, temperature 0, and top-p 1. Shared benchmark validation checks the
model/tokenizer revisions, template hash, actual prompt count, and context
budget. EOS stopping remains enabled; the cap does not guarantee output length.
The workload's `baseline` describes its original CPU experiment, not the
actual device of this run, which is recorded separately under `runtime`.

The remote image pins Torch 2.6.0+cu124, Transformers 4.57.6, NumPy 2.2.6,
Hugging Face Hub 0.36.0, tokenizers 0.22.1, and safetensors 0.6.2 on Python 3.12.
These differ from the earlier local validation stack; that comparison cannot
isolate hardware. The CPU control below matches these six package releases,
but transitive dependencies are not fully locked.

A CPU-only image build step downloads inference artifacts at revision
`7ae557604adf67be50417f59c2c2f167def9a775` into `/opt/huggingface`.
Modal caches that layer; GPU invocations load with `HF_HUB_OFFLINE=1`.
Benchmark helpers are mounted after the download layer so their edits do not
invalidate it. Only the runner, inference module, and required benchmark modules
are included from the project. The workload JSON is passed as an argument;
result directories, virtual environments, local model caches, and credentials
are not uploaded.

With the separate Modal SDK environment installed and authenticated, run from
the repository root **only when ready to consume cloud resources**:

```bash
~/.venvs/modal/bin/modal run modal_inference.py \
  --workload benchmark-results/phase3/workload.json \
  --output benchmark-results/modal/warm-t4-001.json
```

`--workload` defaults to the path above; `--output` is required. Choose a fresh
output path each time. The local entrypoint refuses existing files and uses the
shared exclusive JSON writer, never overwriting prior evidence.

For each measured call, CUDA is synchronized before starting the wall clock
and again before stopping it. `warm_inference_ms` includes formatting,
tokenization, input transfer, prefill, decoding, output transfer, and final text
decoding. It excludes image build/download, container startup, model loading,
warm-ups, network invocation, and time between calls. There is no per-token
instrumentation or synchronization added inside the generation loop.
This is not GPU-kernel-only latency, TTFT, TPOT, or HTTP end-to-end latency.

The saved JSON includes UTC recording time, the workload, actual GPU/device/dtype,
package versions, model revision, template hash, settings, two warm-up rows,
and up to five measured rows. Each row has phase, zero-based iteration, status,
actual prompt/completion counts, and finish reason. Warm-ups have null timing.
The summary reports successful measured median latency and total successful
measured output tokens divided by their summed inference seconds, not the mean
of individual rates or decode-only throughput. It excludes warm-ups and failed
calls, whose counts remain visible. Five samples do not establish a stable p95
or a production SLO. Do not compare this boundary with CPU HTTP E2E latency.

An ordinary generation/synchronization error stops further calls, records its
exception type without private exception text or fabricated completion usage,
and returns earlier rows for saving. An incomplete run exits 1. Ctrl-C handled
inside the local entrypoint saves an interrupted record and exits 130.
If loading, validation, transport, or the remote process fails before results
return, `remote_results_received` is false and `summary` is null: the empty local
sample list does not establish that no remote work ran. Undelivered rows are
unavailable. A forced local termination or image-build failure before the
entrypoint starts cannot be recorded by it; retain the CLI output separately.
Local interruption does not prove that native remote computation has stopped.

This is one ephemeral experiment, with one allowed container, no configured
input retries or minimum warm capacity, and a 180-second function timeout.
The download build step has a separate 600-second timeout. GPU allocation during
loading, warm-ups, generation, and remaining container lifetime consumes
resources. These limits are not hard spending caps; check available credits
and current [Modal pricing](https://modal.com/pricing) before running.

The learner already demonstrated a successful single-completion CUDA run:
Tesla T4, float32, matching revision/template hash, answer `4`, 43 prompt tokens,
two completion tokens, and `stop`. That established functionality, not speed.
The timing loop also passed deterministic boundary/failure tests and a real
CPU Qwen smoke with CUDA synchronization replaced only for local validation.
The learner's first remote timing run is saved in
`benchmark-results/modal/warm-t4-001.json`: two successful warm-ups followed by
five successful measured calls, all producing 128 tokens from 59 prompt tokens.
Median warm inference duration was **3397.13 ms**; 640 measured output tokens
over **17.3633 summed inference seconds** yielded **36.86 output tokens/s**.
There were no warm-up or measured errors.

The earlier local CPU validation observed 9422.45 ms median and 13.31 tokens/s
at the analogous function boundary. The observed median ratio is **2.77x**
(63.95% lower median duration), but this is an exploratory cross-environment
comparison: Torch/Transformers versions, host hardware, and operating systems
differ, and there are only five samples per run. It does not isolate the GPU's
effect, measure HTTP latency or cold starts, or establish a production SLO.
See [the GPU experiment log](logs/practice.md#gpu-exploration-modal-and-warm-inference)
for all measurements and limits.

### Matched-library CPU control

`benchmarks/warm.py` now contains the shared measurement loop formerly in
`benchmarks/gpu.py`. CPU calls run synchronously without touching CUDA;
CUDA calls retain both synchronization boundaries. Other accelerators are
rejected, and the Modal entrypoint still requires CUDA and float32.
The local API, project dependencies, and CPU token instrumentation are unchanged.

The control uses a separate environment, not the project's `.venv` or the Modal
SDK environment. One-time setup on this Apple Silicon Mac:

```bash
.venv/bin/python -m venv ~/.venvs/inference-cpu-control
~/.venvs/inference-cpu-control/bin/python -m pip install \
  torch==2.6.0 transformers==4.57.6 numpy==2.2.6 \
  huggingface-hub==0.36.0 tokenizers==0.22.1 safetensors==0.6.2
```

The macOS Torch wheel has the same release version as the GPU's `2.6.0+cu124`,
but is not the CUDA build. The runner requires offline mode and already-cached
artifacts at the workload's revision. It loads CPU float32 once, sets six
intra-op threads, records the actual inter-op thread count, and executes the
same two warm-ups and five sequential measurements without CUDA mocks.

Recorded invocation from the repository root:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.warm \
  --workload benchmark-results/phase3/workload.json \
  --output benchmark-results/modal/cpu-matched-001.json
```

Output paths are exclusive: choose a fresh path for any separately approved
future run. The CPU JSON records host/thread metadata, all returned rows,
actual counts and finish reasons, and the same whole-function timing summary.
An ordinary sample failure retains earlier rows and stops. If loading,
validation, or interruption prevents the loop from returning, `samples_available`
is false and `summary` is null; an empty list does not prove no local work ran.
Handled failures exit 1 and Ctrl-C exits 130. Forced termination cannot save
a result. This command launches no Modal job.


The approved control completed on 2026-10-28 UTC, using Apple M1 Pro CPU float32,
Python 3.12.13, six intra-op threads and eight inter-op threads. All seven calls
succeeded with 59 prompt tokens, 128 completion tokens, and `length` stopping.

| Metric | Matched-library CPU | Saved T4 run |
|---|---:|---:|
| Measured calls | 5 | 5 |
| Median warm inference | 11604.62 ms | 3397.13 ms |
| Summed measured duration | 58.3692 s | 17.3633 s |
| Aggregate output tokens/s | 10.96 | 36.86 |

The observed CPU/T4 median ratio is **3.416x**, or **70.73% lower T4 median
duration**. The T4/CPU aggregate-throughput ratio is **3.362x**. This reuses
`warm-t4-001.json`; no new cloud run was launched. The CPU median is **23.16%
higher** than the earlier, unmatched CPU validation, not an improvement.

Matching the six core package releases removes that particular confound, not
all software or platform differences: Python patch versions, Torch builds,
OS/architecture, kernels, transitive dependencies, and execution conditions
remain different or uncontrolled. Five samples in one run per environment do
not isolate a hardware-only effect or establish a stable SLO. The results do
not establish why the CPU runs differ or whether their generated text is identical.
See [the matched-control evidence](logs/practice.md#matched-library-cpu-control-2026-10-28)
for every measured duration and the verification commands.

## Static batching: CPU correctness

`src/batched_generation.py` is a standalone Stage 5 experiment, not an API
scheduler change or a performance benchmark. It reuses `load_runtime()`,
`Runtime`, and `GenerationResult`. Its new `generate_batch()` formats each
conversation, left-pads the inputs without changing the tokenizer's shared
padding configuration, checks the context budget, and makes one greedy
`model.generate()` call for the batch.

Results stay in input order. Actual prompt counts come from attention masks;
completion extraction skips the common padded input width. Each result retains
its first generated EOS in the token count and excludes subsequent padding,
including when a padding ID is also an EOS ID. A sequence reaching the output
cap without EOS reports `length`.

The CLI uses the existing matched-library CPU environment and cached revision
`7ae557604adf67be50417f59c2c2f167def9a775`. It loads CPU float32 once with six
intra-op threads, runs the two prompts individually using the unchanged
`src.inference.generate()`, then together as one batch of two.

Recorded invocation:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m src.batched_generation \
  --max-new-tokens 32 \
  --output benchmark-results/batching/cpu-correctness-001.json
```

The approved run completed on 2026-11-02 UTC:

| Input | Actual prompt tokens | Sequential completion tokens | Batched completion tokens | Finish in both |
|---|---:|---:|---:|---|
| `Reply with exactly one word: Hello` | 36 | 10 | 10 | stop |
| Explain inference-server request processing | 59 | 32 | 32 | length |

Both generated token-ID lists matched their individual references exactly.
The first response was `Hello! How can I assist you today?`, not one word:
this is model instruction-following behavior shared by both paths, not a
batching discrepancy. These cases establish mixed-length CPU correctness,
not universal output equivalence, GPU correctness, or improved speed.

The JSON preserves synthetic inputs, generated IDs/text/counts, finish reasons,
runtime metadata, and equality checks. `status=completed` means the comparison
ran; inspect `all_token_ids_equal` separately. Ordinary failures save their type
and any already-returned individual results; interrupted or failed calls have
no fabricated result. Handled failures exit 1 and Ctrl-C exits 130. Existing
output paths are refused before model loading; choose a fresh path for any
future run. Forced termination cannot save evidence.

This correctness script does not measure timing, throughput, or memory. The
separate benchmark below now measures batch sizes 1/2/4 locally; GPU execution
still requires separate approval.
See [the Stage 5 evidence](logs/practice.md#stage-5-static-batching-cpu-correctness)
for verification and the remaining learning goals.

## Static batching benchmark

`benchmarks/batching.py` reuses the existing single-request `generate()` at
batch size 1 and `generate_batch()` at sizes 2 and 4. One loaded model processes
the same four synthetic prompts in fixed input order:

1. Reply with exactly one word: Hello.
2. The inference-server explanation prompt used above.
3. What is 2 + 2? Reply with just the number.
4. Explain in two sentences why attention uses a KV cache during autoregressive decoding.

Each configuration has its own warm-up of the complete workload. One measured
workload means four individual calls, two calls of two prompts, or one call of
four prompts. The model revision, float32 precision, greedy settings, output cap,
and natural EOS policy stay fixed within a run. The first sequential warm-up
supplies the output reference; later warm-ups and measurements retain actual
IDs, text, counts, finish reasons, and reference comparisons. Differences are
reported, not hidden or automatically treated as numerical correctness failures.

### Measurement boundaries

- `workload_ms`: wall time to finish all four prompts, including dispatch,
  per-call timing and loop bookkeeping. Loading, warm-ups, serialization, and
  output comparisons are outside this interval.
- `call_ms`: whole `generate()` or `generate_batch()` call, including tokenization,
  transfers, prefill, decoding, and text decoding. CUDA fences bracket each call;
  an outer fence excludes earlier GPU work from the workload clock.
- Throughput: actual output tokens from complete successful measured workloads
  divided by their summed workload seconds. Partial failed workloads are excluded.
- `median_call_ms` is a median of single-call or batch-call durations, depending
  on configuration. It is not per-request latency or batch time divided by size,
  and calls contain different prompts. Neither timer measures HTTP latency,
  queue wait, TTFT, or TPOT.
- CUDA peak allocated/reserved bytes are absolute PyTorch allocator peaks within
  each measured workload, reset before timing after warm-up. They include model
  weights, not just KV cache, and exclude non-PyTorch allocations. Reserved memory
  can include cached blocks from earlier configurations; no cache flush is used.
  CPU memory fields are null, not zero.

### Completed CPU smoke benchmark

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.batching \
  --device cpu --max-new-tokens 16 --warmups 1 --repeats 2 \
  --output benchmark-results/batching/cpu-benchmark-001.json
```

On 2026-11-02 UTC, both measured workloads per configuration completed. Each
workload generated 44 actual tokens, with counts `[10, 16, 2, 16]` and finish
reasons `[stop, length, stop, length]`. All returned outputs matched the reference.

| Batch size | Calls per workload | Median workload (ms) | Median call (ms) | Aggregate output tokens/s |
|---|---:|---:|---:|---:|
| 1, sequential | 4 | 4483.37 | 1199.98 | 9.81 |
| 2 | 2 | 4154.56 | 2072.48 | 10.59 |
| 4 | 1 | 2193.14 | 2193.14 | 20.06 |

Batch size 4 observed **2.044x** the sequential throughput in this CPU smoke.
Two repeats, short outputs, fixed configuration order, and uncontrolled host
conditions do not establish a stable speedup, p95, production SLO, or GPU result.
These four-prompt/16-cap numbers are not comparable to the earlier one-prompt
59/128 GPU benchmark. Raw workloads, calls, and outputs are preserved locally.

The CLI requires a fresh output path and offline cached weights. It supports
local `--device cpu` or `--device cuda`; CUDA availability and float32/device
placement are checked. Limits: 1-128 output tokens, 1-2 warm-ups, 1-5 measured
workloads per configuration. Defaults are 32/1/3 respectively.

An ordinary failure or handled interruption stops further work. Completed calls
remain visible, the failed call has no fabricated usage/timing, and unstarted
calls are marked `not_run`. Summaries exclude incomplete workloads. Execution
success does not imply zero output mismatches. Forced termination cannot save
local evidence.

### Completed GPU batching run

`modal_batching.py` reuses the existing pinned image and CPU download layer,
mounting only the additional source files after that layer. It leaves
`modal_inference.py` and its original command unchanged.

The learner executed this command and supplied the completed run output:

```bash
~/.venvs/modal/bin/modal run modal_batching.py \
  --max-new-tokens 16 --warmups 1 --repeats 3 \
  --output benchmark-results/batching/t4-benchmark-001.json
```

This requests one T4, 4096 MiB host RAM (not GPU VRAM), at most one container,
zero minimum warm containers, no function retries, and a 300-second function
timeout. The inherited CPU download step has a 600-second timeout. This command
performs 28 generation calls covering 48 completions including warm-ups, with at
most 16 new tokens per completion, matching the CPU smoke's output cap.

These are workload/resource bounds, not a hard spending cap. Image work,
GPU loading, warm-ups, measurements, and remaining container lifetime can consume
resources. Check credits and current [Modal pricing](https://modal.com/pricing)
before approval; stop the app through Modal if abandoning a run.
Local interruption does not prove remote computation stopped. If no result is
returned, `remote_results_received=false`, summaries remain null, and empty
local samples do not mean no remote work happened.

The run recorded at `2026-11-04T00:45:23.031925+00:00` completed on a Tesla T4,
CUDA 12.4, float32. Each configuration completed one warm-up and three measured
workloads, with no failures or output mismatches. Every workload produced
44 actual tokens, with per-input counts `[10, 16, 2, 16]` and finishes
`[stop, length, stop, length]`. The first reference's null comparisons mean
"reference itself", not a failure. All later results matched it.

| Batch size | Median four-prompt workload (ms) | Output tokens/s | Throughput / sequential | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---:|---:|---:|---:|---:|
| 1 | 1062.62 | 41.17 | 1.000x | 1899.75 | 2024 |
| 2 | 927.17 | 47.41 | 1.151x | 1904.42 | 2024 |
| 4 | 465.15 | 94.26 | 2.289x | 1915.43 | 2050 |

Batch size 4 observed **2.289x sequential throughput** and **56.23% lower median
workload duration**. Peak allocated memory increased by **15.69 MiB (0.83%)**;
reserved memory increased by 26 MiB. One set of model weights is shared across
the batch. These total peaks do not isolate KV-cache memory, and retained
allocator reservations are not the same as live tensor allocations.

This verifies real GPU batched execution for these inputs. Three measured
workloads, short contexts/outputs, and fixed configuration order do not establish
a stable SLO, universal speedup, or lower latency for every individual request.
No profiling established which low-level mechanism caused the timing changes.
The CPU and GPU records share prompts/settings and reference IDs, but differ in
platform, Python patch, Torch build, thread settings, and measured repeat count.

Evidence: `benchmark-results/batching/t4-benchmark-001.json` and
[the learner's Modal run](https://modal.com/apps/lenando0307/main/ap-JXmGJQET81U3bwbbZPH2YC).
No assistant confirmation rerun was performed. The path now exists; future
separately approved runs must use a fresh output path.
See [the GPU batching result](logs/practice.md#first-t4-batching-result-2026-11-04).

## Run the local application

From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

`requirements.txt` pins the direct versions used for the recorded results:
torch 2.14.1, transformers 5.18.0, pydantic 2.14.1, fastapi 0.143.3,
uvicorn 0.54.0, and httpx 0.28.1. Transitive dependencies are not locked. The
T4 jobs use a separately pinned image digest and vLLM version.

A clean Python 3.12.13 venv created from these pins installed in 27.5 seconds,
passed all 182 tests, served `/ready`, and completed one real request on
2026-11-08.

The lifespan loads `Qwen/Qwen2.5-0.5B-Instruct` once per application process on
CPU in float32. Model files must be cached locally or downloadable. Importing
`src.api` does not load weights; failed loading aborts startup. Avoid reload
mode and multiple workers when checking model lifetime.

For an explicit model/tokenizer snapshot, call
`src.inference.load_runtime(revision="7ae557604adf67be50417f59c2c2f167def9a775")`.
The keyword-only revision applies to both artifacts; a commit hash pins their
version, not the Python dependencies or device. Omitting it preserves the
default Hugging Face revision behavior. Loading still uses CPU float32.

From another terminal:

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

| Endpoint/state | Status | Body |
|---|---|---|
| `/health`, serving HTTP | 200 | `{"status":"alive"}` |
| `/ready`, initialized and idle | 200 | `{"ready":true,"status":"ready","running":0,"queued":0,"queue_capacity":2}` |
| `/ready`, running with waiting capacity | 200 | `{"ready":true,"status":"busy","running":1,"queued":0,"queue_capacity":2}`; `queued` may also be 1 |
| `/ready`, both waiting slots occupied | 503 | `{"ready":false,"status":"full","running":1,"queued":2,"queue_capacity":2}` |
| `/ready`, uninitialized, if reachable | 503 | `ready:false`, `status:"not_ready"`, plus the same capacity fields |
| `/ready`, shutting down, if reachable | 503 | `ready:false`, `status:"shutting_down"`, plus the same capacity fields |

Readiness describes admission capacity, not whether the model is idle. It is a
snapshot, not a reservation. Uvicorn may refuse connections until startup
completes; connection refusal is not an HTTP 503 response.

Press Ctrl-C to stop normally. Uvicorn closes its listener and drains admitted
HTTP requests before lifespan teardown. At teardown, the application rejects new
work, removes any remaining queued jobs, signals running cancellation, and waits
off the event loop for the actual worker to exit before releasing the runtime.
A cancellation flag cannot interrupt a stuck native call; such a call can still
prevent graceful shutdown and require process termination.

### Optional bearer authentication

`INFERENCE_API_TOKEN` gates `POST /v1/chat/completions` and `GET /metrics`.
Unset or empty leaves both open, which is the documented mode for loopback use.

```bash
INFERENCE_API_TOKEN="$TOKEN" .venv/bin/python -m uvicorn src.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

`/health` and `/ready` remain unauthenticated for probes. A missing or wrong
token returns 401 with `WWW-Authenticate: Bearer`; the token is never logged.
Authentication runs before request logging, so unauthorized attempts produce no
lifecycle log and no `x-request-id` header. They are counted in `/metrics` as
`errors.unauthorized`.

### Metrics

`GET /metrics` returns one JSON snapshot from the current process. Counters reset
on restart; nothing is aggregated across instances or processes.

| Field | Meaning |
|---|---|
| `requests` | HTTP outcomes: `completed`, `rejected`, `failed`, `disconnected` |
| `errors.<code>` | count per sanitized error code, for example `invalid_request` |
| `queue` | live `capacity`, `queued`, and `in_flight` |
| `latency_ms` | last 256 HTTP request durations with `p50`, `p95`, `max`, `last` |
| `tokens` | prompt/completion totals; rate uses producer wall time only |
| `memory.process_peak_rss_bytes` | process high-water RSS, including the model; not GPU memory |
| `uptime_seconds` | seconds since this process initialized |

If startup has not completed, `/metrics` returns 503 `not_ready`. The endpoint
never includes prompts, outputs, headers, or exception messages.

```json
{
  "queue": {"capacity": 2, "queued": 0, "in_flight": 0},
  "requests": {"completed": 1, "rejected": 0, "failed": 0, "disconnected": 0},
  "errors": {},
  "latency_ms": {"count": 1, "p50": 1187.61, "p95": 1187.61, "max": 1187.61},
  "tokens": {"prompt": 36, "completion": 4, "completion_per_second": 3.377},
  "memory": {"process_peak_rss_bytes": 2127249408}
}
```

## Operations and recovery

Verified on this machine with the cached CPU checkpoint:

- **Restart.** SIGTERM drained the listener, logged `Application shutdown
  complete`, and left no listener on the port. A second process started normally
  with fresh counters (`uptime_seconds` near zero, `requests.completed` zero).
  Uvicorn 0.54 re-raises a captured SIGTERM after graceful shutdown, so the
  process status is `-15` even though teardown completed; check the shutdown log
  lines rather than an expected exit code of zero.
- **Failed startup.** Starting with an empty `HF_HOME` plus
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` aborted before serving:
  `Application startup failed. Exiting.`, exit code 3, and no listener. The
  service never answered `/ready`.
- **Clean environment.** A Python 3.12.13 venv created from the pinned
  `requirements.txt` installed in 27.5 s, passed 182 tests, answered `/ready`,
  and completed one real request. Transitive versions still resolve freely.
- **Model cache.** Weights and tokenizers must already be cached or downloadable
  at startup; `HF_HOME` selects the cache. Offline evaluation requires a
  pre-populated cache. Loading happens once per process in the lifespan, and a
  loading failure releases the runtime and stops startup.
- **Credentials.** Supply the bearer token through the environment, never on the
  command line or in committed files. The default bind is loopback; add
  authentication, network limits, and TLS termination before exposing the
  service beyond a trusted host.
- **Resource bounds.** `max_tokens` is capped at 512, prompt-plus-output at the
  model context limit, completion bodies at 64 KiB, the queue at two waiting jobs,
  and one worker runs one generation at a time.

## Request an ordinary chat completion

With the server running, use another terminal:

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is 2 + 2? Reply with only the number."}
    ],
    "max_tokens": 16,
    "temperature": 0,
    "top_p": 1,
    "stream": false
  }'
```

This is a supported chat-completions subset, not full OpenAI API compatibility.

| Field | Rule | Default |
|---|---|---|
| `model` | Exactly `Qwen/Qwen2.5-0.5B-Instruct` | Required |
| `messages` | Nonempty array; roles `system`, `user`, `assistant`; nonblank string content | Required |
| `max_tokens` | Integer from 1 through 512, counting new output tokens | 100 |
| `temperature` | Finite number from 0 through 2; zero selects greedy decoding | 0 |
| `top_p` | Finite number greater than 0 and at most 1; must be 1 for greedy decoding | 1 |
| `stream` | Boolean; true selects incremental SSE, false selects ordinary JSON | false |

Numeric strings, booleans used as numbers, and unknown request/message fields
are rejected. Tools, images, `n`, `top_k`, `stop`, `seed`, and `stream_options`
are unsupported. Valid message text is preserved. The client supplies history
on every request; the server does not retain conversations.

A successful response is HTTP 200 with `object: "chat.completion"`, an ID
prefixed with `chatcmpl-`, an integer creation timestamp, the model ID, and one
choice containing `message.role`, `message.content`, and `finish_reason`.
The finish reason is `stop` for EOS or `length` for an exhausted output cap.
`usage` contains actual `prompt_tokens`, `completion_tokens`, and `total_tokens`.
Completion counts include generated EOS IDs, even when those IDs are invisible
in the decoded text.

Every handled chat success/error includes a server-generated `X-Request-ID`;
incoming request identifiers are not trusted. Errors use
`{"error":{"code":"...","message":"...","request_id":"..."}}`, with the same ID
as the response header. Validation responses do not echo inputs or arbitrary
field names; unexpected failures do not expose exception text or tracebacks.

| Status | Error code | Meaning |
|---|---|---|
| 422 | `invalid_request` | Invalid JSON, schema, types, ranges, model, or fields |
| 422 | `invalid_parameter_combination` | Greedy decoding with non-default top-p |
| 400 | `context_length_exceeded` | Formatted prompt plus requested output exceeds the supported context |
| 429 | `server_busy` | One producer is active and both FIFO waiting slots are occupied |
| 503 | `not_ready` | Application unavailable or shutting down |
| 503 | `request_canceled` | Generation was canceled before response headers |
| 503 | `slow_consumer` | Streaming could not keep up before response headers |
| 500 | `generation_failed` | Submission, preparation, or generation failed unexpectedly |

The combined budget is at most 2048 tokens, further bounded by the model's
declared context. Exact fits are accepted; excess input is rejected without
truncation, dropping history, or reducing the requested cap. Normal unknown
routes and wrong methods retain HTTP 404/405 behavior.

## Stream a chat completion

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "List five animals."}],
    "max_tokens": 64,
    "stream": true
  }'
```

The response uses `Content-Type: text/event-stream`, `Cache-Control: no-cache`,
and `data: <JSON>\n\n` frames. Each completion chunk has the same ID, model,
creation timestamp, and `object: "chat.completion.chunk"`:

1. Initial chunk: `choices[0].delta` is `{"role":"assistant"}`.
2. Nonempty content chunks: newly decoded text in `choices[0].delta.content`.
3. Successful terminal chunk: empty delta, actual `finish_reason`, and top-level
   `usage`. Earlier chunks have a null finish reason and no usage.
4. Exactly one `data: [DONE]\n\n` frame ends a connected successful stream.

Terminal usage is a **local extension**; `stream_options` remains unsupported.
Text chunks are not tokens. Counts come from generated IDs, including EOS,
using the same generation path and decode settings as ordinary responses.

Formatting, tokenization, and context validation finish before SSE headers.
Failures known before the response starts use the ordinary HTTP error table.
After headers, a producer failure retains HTTP 200 and sends the error envelope
as SSE with code `generation_failed`, followed by `[DONE]`, if still connected.
It does not send a successful finish reason or fabricated usage.

### Execution ownership

Call `submit_inference(app, validated_request)` in `src/api.py` on the
application's event loop with an already validated `ChatCompletionRequest`.
It returns an internal job with a `result` future, cancellation token, timestamps,
and scheduling state. It raises `InferenceUnavailable` when unavailable and
`InferenceBusy` only when the two waiting slots are full.

The event loop owns the FIFO queue. Only the active job enters the single-worker
executor; waiting jobs do not accumulate in the executor's hidden queue. The
worker performs formatting, tokenization, context validation, and generation.
Its actual completion callback releases ownership and starts the oldest waiting
job, even after an error or abandoned delivery.

Job transitions are `queued -> running -> completed/failed/canceled`, or
`queued -> canceled`. Context rejection and failed submission are scheduling
failures; delivery/error codes retain their more specific meanings. There is no
queue TTL or overall inference deadline.

Use `cancel_inference(app, job, reason)` for internal cancellation. Removing a
queued job immediately frees waiting capacity without tokenizing or running it.
A running job receives a cooperative signal, checked around preparation and
between generation steps through a Transformers stopping criterion. HTTP
cancellation does not cancel the retained producer future or free the worker
early. Abandoned future exceptions are observed without logging their text.

Disconnected clients are not promised a terminal event or `[DONE]`. Cancellation
is not a successful `stop`/`length` completion and does not fabricate usage.
A late delivery failure cannot relabel already completed producer work.

`src/streaming.py` preserves Transformers' incremental text decoding but bounds
the queue to eight items, including its terminal sentinel. Empty text increments
do not consume capacity. A full queue waits at most one second, checking
cancellation at intervals no longer than 50 ms, then cancels as `slow_consumer`.
Each ASGI chat send has a five-second deadline. A writable stream can send a safe
error event and `[DONE]`; a blocked transport is terminated without that promise.
Uvicorn may report an incomplete response when it closes such a transport.

Generation configuration and stream buffers are per request. `seed` remains
unsupported; sampling uses the process RNG in serialized execution order.
The server does not reset global random state to emulate per-request seeds.

## Private request lifecycle logs

The normal Uvicorn command also emits one JSON line per lifecycle event through
Python's `inference.lifecycle` logger at INFO level, to stderr. No log files or
external telemetry service are created. Uvicorn's own startup/access logs remain
separate and may be plain text.

Logging covers `POST /v1/chat/completions`. A server-generated UUID is assigned
before body parsing and validation and reused in `X-Request-ID`, completion IDs,
error envelopes, and every lifecycle event. Incoming identifiers are ignored.

| Event | Meaning |
|---|---|
| `request_accepted` | A running or queued job was admitted; submission/generation may still fail |
| `request_started` | The worker actually entered the job, with its measured queue wait |
| `request_finished` | Delivery completed, was rejected/failed, or ended on a detected disconnect/cancellation |
| `producer_finished` | A started worker exited, even if its client already disconnected |
| `scheduling_finished` | Every admitted job reached a terminal state, including queued cancellation and submission failure |

Each event includes `request_id`, the configured `model`, `stream`, and
`http_status`. Unknown stream mode or HTTP status is `null`, not guessed.
Invalid client-supplied model names are never logged. Delivery reports
`completed`, `rejected`, `failed`, or `disconnected`. Producers report `completed`,
`rejected`, `failed`, or `canceled`. `scheduling_finished.state` is `completed`,
`failed`, or `canceled`, independently of delivery.

Validation, full-queue, and availability rejections have a `request_finished`
event without admission/scheduling events. Context rejection happens in an
admitted worker, so it also has acceptance, producer, and scheduling events.
Submission failure and queued cancellation have scheduling but no producer
event. A stream can have HTTP 200 with failed delivery because headers were
already sent.

Timing fields use a monotonic clock:
- `request_finished.duration_ms` starts before validation/admission and ends after
  the final ASGI body send completes, or when disconnect/cancellation is observed.
  Successful send completion does not prove receipt by the remote client.
- `producer_finished.producer_duration_ms` measures the worker separately,
  including formatting, tokenization, context validation, generation, and output
  extraction. It excludes request-body reading and executor handoff.
- `request_started.queue_wait_ms` measures validated admission to actual worker
  entry, separately from model TTFT. Terminal records use null for jobs canceled before
  starting.
- `scheduling_finished.cancellation_delay_ms` measures the first cancellation
  signal to actual worker exit. It is null for never-started jobs and jobs without
  an in-flight cancellation signal. Signals after worker exit are delivery-only.
  The first cancellation reason and timestamp win.

Delivery may end before authoritative scheduling and cancellation measurements
exist. Correlate the later scheduling/producer events using the same request ID.
These durations are lifecycle measurements, not TTFT/TPOT benchmarks.

Actual `prompt_tokens` and `completion_tokens` are included when a successful
producer result is available. Otherwise the fields are omitted, not replaced by
zero, the requested cap, or estimated partial counts. A disconnected request may
therefore have counts only in its later producer event.

Errors include a sanitized `error_code`; unexpected failures also include
`exception_type`, never the exception message. Lifecycle records exclude messages,
rendered prompts, generated text, token IDs, raw bodies, authorization headers,
raw validation inputs, and tracebacks. The middleware observes send/receive
boundaries without buffering or logging response bodies.

To inspect a request, send either curl example above with `-i`, then find its
`X-Request-ID` in the server's JSON lines. Do not interpret Uvicorn's streaming
`200 OK` access line as completed generation; inspect `request_finished.outcome`
and, after a disconnect, the separate `producer_finished` event.

## Reproducible benchmarks

Phase 3 keeps the CPU float32, single-worker service as its baseline. Benchmark
commands are separate modules under `benchmarks/`; normal `src.api:app` behavior
and its HTTP contract are unchanged. No additional packages are required.

### Create the fixed workload

Run from the repository root. Model artifacts must already be in the Hugging
Face cache: these tools prohibit downloads during measured startup. With the
verified snapshot, formatted inputs contain 59 or 990 tokens, paired with output
caps of 32 or 128. Exact messages, template hash, actual counts, and snapshot
revision are saved in the manifest:

```bash
.venv/bin/python -m benchmarks.workloads \
  --output benchmark-results/run-1/workload.json
```

These are synthetic prompts, not a quality evaluation. The long input repeats
background material to vary context length without changing the task. All cases
fit the service's 2048-token combined context budget.

### Measure HTTP delivery and model token production

In terminal A, choose an unused localhost port, such as 8001:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m benchmarks.server --port 8001 \
  --metrics benchmark-results/run-1/server.jsonl
```

This opt-in server loads the actual runtime once and preserves production
scheduling and shutdown. It records model timing for executed generations and
safe terminal scheduling metadata, including jobs canceled before starting.
Scoped wrappers correlate rows through the server-generated `X-Request-ID` and
restore original methods at shutdown. No extra inference worker is introduced.

After startup completes, run terminal B:

```bash
.venv/bin/python -m benchmarks.client \
  --url http://127.0.0.1:8001 \
  --workload benchmark-results/run-1/workload.json \
  --warmups 2 --repeats 20 \
  --output benchmark-results/run-1/http.json

.venv/bin/python -m benchmarks.report \
  --client benchmark-results/run-1/http.json \
  --server-metrics benchmark-results/run-1/server.jsonl \
  --output benchmark-results/run-1/report.json
```

The default client concurrency is one. `--concurrency N` issues bounded waves of
at most N requests; `--repeats` remains the total measured request count per case,
not a count per client. Each case has separate sequential warm-ups and its own
measured window. Errors and HTTP 429 rejections are recorded without retries.
The client saves evidence before returning nonzero for unexpected failures or
rejections. Ctrl-C saves collected samples and returns 130.

Compare concurrency without changing the manifest, repetition count, server, or
generation settings. Run these sequentially, with the same instrumented server:

```bash
.venv/bin/python -m benchmarks.client --url http://127.0.0.1:8001 \
  --workload benchmark-results/run-1/workload.json --warmups 2 --repeats 20 \
  --concurrency 1 --output benchmark-results/run-1/http-c1.json
.venv/bin/python -m benchmarks.client --url http://127.0.0.1:8001 \
  --workload benchmark-results/run-1/workload.json --warmups 2 --repeats 20 \
  --concurrency 2 --output benchmark-results/run-1/http-c2.json
.venv/bin/python -m benchmarks.client --url http://127.0.0.1:8001 \
  --workload benchmark-results/run-1/workload.json --warmups 2 --repeats 20 \
  --concurrency 4 --output benchmark-results/run-1/http-c4.json
```

With one running plus two waiting slots, a simultaneous wave of four can reject
one request. That is expected admission control, not a reason to retry it during
the measurement. Generate a report for each client file using the same server
JSONL. Record success counts alongside latency percentiles and useful throughput.

For a separate cancellation control, add `--cancel-after-text`: measured streams
close after their first real content event, while warm-ups complete normally.
These intentional cancellations are not CLI failures and have no completion
usage. Wait for server scheduling completion before joining cancellation-delay
metrics. A client disconnected before headers has no response request ID;
server logs still account for it, but client/server correlation is unavailable.

Stop the benchmark server normally with Ctrl-C before running model-side
experiments; do not benchmark competing model processes simultaneously.

The client can target another compatible server by changing `--url` and supplying
`--server-config path/to/config.json`. That JSON object is declared configuration,
not hardware detected by the client. A compatible server must provide valid
terminal usage and `[DONE]`; a separate final usage chunk is also accepted.
Without model instrumentation, omit `--server-metrics`: model timings and server
memory remain unavailable rather than being estimated from text chunks.

### Natural stopping, fixed-length generation, and profiling

Natural HTTP runs preserve EOS and record actual output lengths, which may be
shorter than their caps. The separate model-side harness can run natural or
fixed-length experiments:

```bash
.venv/bin/python -m benchmarks.model \
  --workload benchmark-results/run-1/workload.json \
  --mode fixed --warmups 2 --repeats 20 \
  --output benchmark-results/run-1/fixed.json
```

Use `--mode natural` for a standalone natural-stopping comparison, and repeat
`--case CASE_NAME` to select cases. Fixed mode sets matching `min_new_tokens` and
`max_new_tokens` only on the fresh per-call generation configuration. It suppresses
ordinary EOS until the requested length without changing shared defaults,
production API fields, or context/output caps. It is not a quality measurement.

Profile a representative slow case in a separate process/run:

```bash
.venv/bin/python -m benchmarks.model \
  --workload benchmark-results/run-1/workload.json \
  --profile-case long-input-long-output --warmups 2 \
  --output benchmark-results/run-1/profile.json
```

Profiling produces operator self/total CPU times, call counts, allocation
statistics, and a neighboring `.trace.json` Chrome trace. These instrumented
timings are excluded from the repeated-run headline statistics.

### Interpret the measurements

- **Client E2E:** request start to receipt of `[DONE]`, not eventual connection
  closure. A malformed or incomplete stream is an error, not a successful sample.
- **Client first text:** request start to first nonempty content event; role-only
  events do not count.
- **Opt-in engine capture:** the continuous-batching runner retains generated text,
  returned token IDs, and stream-event timestamps. Event gaps are delivery-chunk
  gaps, not necessarily per-token latency. The ordinary client still omits text
  and event capture unless explicitly requested.
- **Model TTFT:** immediately before `model.generate()` to its first generated
  token callback. Includes generation setup/prefill, excludes our tokenization.
- **Queue wait:** validated admission to actual worker entry. Client first-text
  and E2E include this wait; model TTFT does not.
- **Cancellation delay:** first server cancellation signal to worker exit,
  not the client's socket-close duration. Queued cancellations have no worker
  delay. Missing or historical metrics remain unavailable, never assumed zero.
- **Mean TPOT:** `(last_token_time - first_token_time) / (N - 1)`, unavailable for
  zero or one output token. Inter-token intervals are reported separately.
- **Useful output throughput:** successfully delivered generated IDs divided by
  the complete measured case window, including failure time and client overhead.
  EOS is counted when generated. Failed or canceled work is not useful output.
- **Memory:** `process_peak_rss_bytes` is process lifetime high-water RSS, not
  current memory, per-request allocation, or Apple unified/device memory.

Reports join server/client durations by request ID, never by subtracting clocks
across processes. They reject mismatched model revisions and token accounting.
Warm-ups, startup/loading, profiling, and measured requests remain separate.
Hardware, package versions, model revision, actual generation settings, and
PyTorch thread counts accompany the results. The recorded baseline launch uses
`HF_HUB_OFFLINE=1` and `TOKENIZERS_PARALLELISM=false`.

Twenty samples per case make p95 exploratory, not a stable production tail
estimate. Synthetic workload order, background activity, thermal state, and
instrumentation overhead can affect comparisons. No device/concurrency
optimization should be justified from a single run.

Artifacts live under ignored `benchmark-results/`. Commands refuse to overwrite
existing result files; choose a new run directory when repeating an experiment.
Measured evidence and bottleneck interpretation are recorded in
[`logs/practice.md`](logs/practice.md).

## Run behavioral tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers validation, context budgets, FIFO capacity and promotion,
readiness, queued removal, retained worker ownership, cooperative cancellation,
bounded-buffer pressure, send deadlines, shutdown, and request isolation.
It also covers SSE progress/terminal metadata, preparation/producer failures,
and lifecycle log privacy, correlation, and timing boundaries. Tiny offline
models exercise real decoding, cancellation, context limits, and EOS-at-cap
precedence, including invisible EOS token accounting. Controlled ASGI connections
expose frames and send pressure without HTTPX response buffering.

Real-Qwen HTTP and controlled-ASGI smoke evidence is recorded separately in
`logs/practice.md`. Benchmark regressions cover bounded waves, equal request
counts, no retries, interruption cleanup, malformed/truncated SSE, usage
accounting, authoritative scheduling joins, historical metric availability,
and fixed-length configuration isolation.


## Inspect tokenization and two-token generation

Run from the repository root in the project environment:

```bash
python src/main.py
```

This script loads `Qwen/Qwen2.5-0.5B-Instruct` in float32 using Transformers
and inspects the steps behind next-token prediction without `model.generate()`.

- Prints the raw prompt's token pieces, IDs, and decoded text.
- Applies the chat template before preparing the model input.
- Prints input ID, embedding table, and input embedding shapes.
- Shows the model's hidden dimension, query heads, key/value heads, and layer count.
- Runs a forward pass with `use_cache=False` and inspects final-position logits.
- Applies softmax and displays the five highest-probability next-token candidates.
- Selects one token with greedy decoding, appends it and extends the attention mask,
  then processes the entire extended sequence to select a second token.
- Prints the two generated tokens together.

The top-five display is inspection only, not top-k sampling. For a prompt of
length `S`, the two forward passes process `S` and `S + 1` positions respectively.

## Inspect embeddings and attention projections

```bash
python src/architecture.py
```

This script loads the same Qwen model in float32 and inspects the beginning of
its forward computation rather than generating text.

- Prepares chat-formatted input and looks up the token embeddings.
- Prints input ID, embedding table, and input embedding shapes.
- Shows the hidden dimension, query heads, key/value heads, and layer count.
- Applies the first Transformer layer's input normalization to the embeddings.
- Applies that layer's `q_proj`, `k_proj`, and `v_proj` to the normalized vectors
  and prints the resulting query, key, and value shapes.

The printed projections are before reshaping into attention heads and applying
rotary positional embeddings to queries and keys. The script does not compute
attention scores, run the full Transformer, or create a KV cache.

## Uncached autoregressive generation

```bash
python src/uncached_generation.py
```

At the terminal prompt, choose:

- `1`: greedy decoding, selecting the highest-scoring token with `argmax`.
- `2`: sampling, dividing logits by temperature (`0.7`), applying softmax,
  then selecting a token with `torch.multinomial`.

The script uses the same Qwen model in float32 and prepares a chat-formatted
prompt. After prefill, it selects a token and, if generation continues, appends
it to the input IDs and extends the attention mask. Each subsequent forward
processes the entire growing sequence with `use_cache=False`.

Generation stops at a configured EOS token ID or `max_new_tokens` (currently 100).
It reports prompt forward-pass time, subsequent-generation time, actual tokens
generated after the first, and the stop reason. Both timers use the same
boundaries as the cached script below, following an untimed prompt warm-up.
There is no per-token printing inside the timed loop. The final response decodes
only the generated IDs, excluding the prompt.

Sampling currently has no top-k or top-p filtering and no fixed random seed,
so sampled responses can differ between runs.

## Cached greedy generation

Run `python src/cached_generation.py` in the project environment.

The script processes the prompt once (prefill), then passes only the newest
selected token and the existing KV cache into each decode forward pass.
It prints prompt tensor shapes and cache information, then decodes the generated
IDs together into the final response. There is no per-token printing in the loop.

Generation stops at a configured EOS token or `max_new_tokens` (currently 100).
The last selected token is not fed back into the model after stopping, so the
final cache length is prompt length + generated token count - 1.
Keep `src/uncached_generation.py` as the uncached comparison, with matching prompts
and greedy settings. Matching tokens checks generation mechanics, not factual
accuracy or speed.

The module is import-safe. `generate_cached(model, model_inputs, max_new_tokens=...,
pad_token_id=..., cache_capacity=None, fixed_length=False)` also accepts left-padded batches, handles each row's EOS
independently, and returns generated IDs, finish reasons, and prefill/final cache
snapshots. Rotary positions follow each row's attention mask. Dense cache rows
are not reclaimed when a sequence finishes: batch cache length is padded prompt
width + the longest generated token count - 1, including EOS in that count.
Omitting `cache_capacity` keeps dynamic allocation. Providing an integer uses
the installed `StaticCache`; capacity must cover every possible processed slot
(`prompt_width + max_new_tokens - 1`) and stay within the model context limit.
Warm-up and measured generation use separate caches. The library builds attention
masks for the chosen cache, excluding unused static positions.
For fixed-work speed experiments only, `fixed_length=True` suppresses every
configured EOS logit until the output cap. It does not change model generation
settings; the default and subsequent natural-EOS calls retain their stopping behavior.
On CUDA this experiment helper also resets the device's peak-memory counters
after its discarded warm-up and returns synchronized allocated/reserved peaks.
Do not run other CUDA workloads concurrently when interpreting those counters.

On CPU, the script reports one prompt forward-pass time after an untimed prompt
warm-up, whose outputs and cache are discarded. It separately times subsequent
generation, starting after selecting token 1 and ending after selecting the last
token and checking its stopping condition. For 100 output tokens, that interval
covers 99 decode forward passes and their token selections, plus attention-mask
updates and Python loop overhead. Loading, tokenization, diagnostic printing,
and final text decoding are outside both timed intervals.

These are single-run observations, not repeated benchmark estimates. The prompt
forward-pass time is not full time to first token, and the prompt-only warm-up
does not separately warm up the single-token decode path.

## Dynamic KV-cache scaling

Run from the repository root with the matched CPU environment and cached model
artifacts. The command loads the same pinned Qwen revision as the batching
experiments, uses float32 and six intra-op threads, and never launches a GPU job:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.cache_scaling \
  --max-new-tokens 16 \
  --output benchmark-results/cache-scaling/cpu-dynamic-001.json
```

Choose a fresh output filename when repeating; existing evidence is never
overwritten. The output cap defaults to 16 and accepts integers from 1 to 128.
Natural EOS can stop a sequence earlier.

The five cases vary one dimension at a time:

- Batch 1 with a fixed sentence repeated 1, 8, and 32 times before the same
  question: 52, 143, and 455 chat-formatted tokens with the pinned tokenizer.
- Batches 2 and 4 using the same 143-token prompt in every row. These batches
  need no padding and do not share cache tensors across identical prompts.

Each case runs the existing manual loop, then compares its IDs, prompt counts,
and finish reasons against Hugging Face `generate()` through `generate_batch`.
The JSON records actual per-row counts, cache shapes, dtype, cached positions,
and both predicted and measured logical K/V bytes after prefill and final decode:

```text
predicted bytes = 2 * layers * batch * KV heads * cached positions * head dimension * bytes/element
measured bytes  = sum(tensor.numel() * tensor.element_size() for every K and V tensor)
```

The prediction uses model configuration and the observed cache dtype/length.
Checks also verify that cache growth counts processed positions, not the final
selected token. A mismatch returns failure and preserves the evidence; execution
errors or interruption preserve completed case data and stop remaining work.

These logical tensor bytes are not CPU RSS or GPU allocator measurements.
Allocated/reserved GPU peaks remain `null` on CPU. Timings are diagnostic
single-run values, not throughput comparisons; the prefill warm-up and reference
generation are outside both timers. The CPU-only static comparison below is a
separate opt-in mode; the default five-case dynamic study is unchanged.
CPU and GPU results and limitations are recorded in
[the Stage 6 evidence](logs/practice.md#stage-6-dynamic-kv-cache-scaling).

### Completed GPU cache-scaling run

`benchmarks.cache_scaling` accepts `--device cpu|cuda` (CPU by default). CUDA
requests fail rather than fall back to CPU when CUDA is unavailable. Both paths
require float32 and keep the same five cases, revision, and correctness checks.

`modal_cache_scaling.py` reuses the existing pinned T4 image and CPU model-download
layer. Only the additional source modules are mounted; local result directories
are not uploaded. Importing the module and displaying help submit no remote work
and require no local Torch installation.

The learner executed this command and supplied the completed result:

```bash
~/.venvs/modal/bin/modal run modal_cache_scaling.py \
  --max-new-tokens 16 \
  --output benchmark-results/cache-scaling/t4-dynamic-001.json
```

This requests one T4, 4096 MiB host RAM (not GPU VRAM), at most one container,
zero minimum warm containers, no function retries, and a 300-second function
timeout. The inherited CPU model-download step has a 600-second timeout.
There are five manual generations, five HF reference generations, and five
discarded prefill warm-ups. The output cap of 16 matches the original
CPU study; it is validated locally before remote submission.

These bounds are not a hard spending cap. Image work, GPU model loading, warm-ups,
generation, and remaining container lifetime can consume resources. Check current
[Modal pricing](https://modal.com/pricing) and credits before approval. A local
interruption does not prove remote computation stopped; stop the app through
Modal if abandoning it. If no result returns, `remote_results_received=false`
means no remote evidence was received, not that no remote work occurred.

For each CUDA case, the existing manual helper:

1. Runs prefill warm-up, releases its outputs/cache, then synchronizes CUDA.
2. Resets peak allocated/reserved counters before measured prefill.
3. Runs prefill and decode with synchronized timing boundaries.
4. Synchronizes and captures both peaks after manual generation and result
   preparation, before returning to the runner.
5. The runner preserves these peaks, then executes the HF correctness reference.

The per-case peaks are **absolute PyTorch allocator bytes**, including weights,
K/V, logits, and temporary tensors, not KV-only bytes or total device usage.
No `empty_cache()` is used. Reservations from warm-ups and earlier cases,
including their reference generations, can remain in subsequent reserved peaks.
Resetting peak counters does not release reservations. The current case's
reference allocation is excluded from its measured interval.

The result recorded at `2026-11-04T18:56:43.964478+00:00` completed on a
Tesla T4, CUDA 12.4, float32, with `remote_results_received=true`.
All five cases passed all six checks. Each of nine manual completions selected
16 tokens and matched its same-device HF reference. Comparison of the saved
files also found exact CPU/GPU token-ID and logical-cache snapshot agreement.
Cross-device output equality is an observation for these cases, not a guarantee.

| Batch | Prompt positions per row | Final logical KV (MiB) | Peak allocated (MiB) | Peak reserved (MiB) |
|---:|---:|---:|---:|---:|
| 1 | 52 | 1.5703 | 1925.80 | 2056 |
| 1 | 143 | 3.7031 | 1981.16 | 2142 |
| 1 | 455 | 11.0156 | 2173.02 | 2414 |
| 2 | 143 | 7.4063 | 2068.92 | 2416 |
| 4 | 143 | 14.8125 | 2244.92 | 2752 |

The final KV column is a snapshot; allocator columns are maxima over the manual
run, not necessarily taken at that snapshot or at the same instant as each other.
Do not subtract these columns to infer weight size or instantaneous unused memory.
The drop from 2173.02 to 2068.92 MiB allocated while reserved stays near 2414 MiB
illustrates why retained reservations are not the same as live tensor demand.
It does not establish a leak or identify an exact allocation breakdown.

This is one observation per fixed-order case, not a stable performance benchmark.
The original 16-token CPU study is the matching control, not the later two-token
compatibility smoke. No experiment was rerun to record this supplied result.
The local artifact is `benchmark-results/cache-scaling/t4-dynamic-001.json`;
choose a fresh output path and obtain separate approval for any additional run.
See the [Modal run](https://modal.com/apps/lenando0307/main/ap-3u0we2FS5gqQLYux4Deuy2)
and [recorded T4 cache evidence](logs/practice.md#first-t4-dynamic-cache-result-2026-11-04).

## Dynamic versus static KV-cache allocation

The same decoder now supports a bounded CPU comparison of dynamic allocation
against static capacities of 160 and 256 positions. This changes cache memory
allocation, not batching: every case uses batch size 1, the same 143-token
chat-formatted prompt, the pinned model, float32, and greedy decoding.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.cache_scaling \
  --compare-static --device cpu --max-new-tokens 16 \
  --output benchmark-results/cache-scaling/cpu-allocation-001.json
```

The recorded file already exists; choose a fresh output path for another run.
`--compare-static` accepts output caps from 1 to 16 and rejects CUDA selection
before loading a model. Without the flag, the existing dynamic CPU/CUDA study
and its 1..128 output-cap range remain available. Modal commands are unchanged
and still run only the dynamic study.

The three-case run completed on CPU. All cases generated the same 16 token IDs,
matched Hugging Face generation, and passed the allocation/capacity checks:

| Cache | Processed positions, prefill → final | Tensor capacity, prefill → final | Prefill K/V MiB | Final K/V MiB |
|---|---:|---:|---:|---:|
| Dynamic | 143 → 158 | 143 → 158 | 3.3515625 | 3.703125 |
| Static, 160 | 143 → 158 | 160 → 160 | 3.7500000 | 3.750000 |
| Static, 256 | 143 → 158 | 256 → 256 | 6.0000000 | 6.000000 |

These library versions allocate the full static backing tensors lazily on the
first forward pass, then update them in place. Unused capacity occupies tensor
storage but is not valid attention history. At the final snapshot, capacity 160
has 2 unused slots (48 KiB); capacity 256 has 98 unused slots (2.296875 MiB).

Snapshot fields distinguish:

- `cached_positions`: processed cache slots, including dense batch padding.
- `reported_cache_length`: the library's raw length query, retained as a diagnostic.
- `allocated_positions`: the actual K/V tensor sequence dimension.
- `predicted_used_bytes`: the model formula evaluated at processed positions,
  not an additional allocation or a physical-memory measurement.
- `predicted_bytes`: expected tensor storage using configured static capacity
  or dynamic processed length.
- `measured_bytes`: the actual sum of all K/V `numel() * element_size()`,
  including unfilled static positions.

Cache write positions and per-row rotary positions are separate. The decoder
passes an explicit write cursor and tracks successful forwards. In the pinned
Transformers 4.57.6, the static-cache raw length query counts nonzero keys and
can undercount zero-valued padding or valid zero keys; it is not used to infer
our write cursor. Attention masks continue to cover only the processed prefix.

The comparison ran eagerly, without `torch.compile`, a custom allocator, or
a GPU job. Single-run timings are diagnostic, not evidence that either cache
strategy is faster. Logical tensor storage is not process RSS or GPU allocator
reserved memory. See [the allocation evidence](logs/practice.md#cpu-dynamicstatic-cache-allocation-2026-11-05)
for exact bytes, runtime metadata, checks, and limitations.

## CPU precision and quantization

`benchmarks/precision.py` compares the pinned Qwen checkpoint on the same CPU
using FP32, FP16, and PyTorch dynamic INT8 with QNNPACK. It installs nothing,
does not launch a GPU job, and leaves the production loader and existing FP32
benchmark guards unchanged.

Recorded command, using the existing matched CPU environment:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.precision \
  --output benchmark-results/precision/cpu-precision-001.json
```

That artifact exists; choose a fresh path for another run. Offline mode is
required. Caught failures/interruptions retain completed results in a failed or
interrupted artifact and return a nonzero exit status.

Speed uses batch 1, six CPU intra-op threads, greedy decoding, the existing
59/990-token prompts, and exactly 32 selected tokens with EOS suppressed.
Each case has one discarded full generation and one measured generation;
the cached decoder additionally discards a prefill warm-up on each invocation.
Loading, conversion, tokenization, cache inspection, and serialization are not
timed. Prefill measures the prompt forward; decode measures the subsequent
31 forwards/selections after selecting token 1. This is not HTTP or end-to-end
generation latency.

Single-run observations on the Apple M1 Pro, Torch 2.6.0 / Transformers 4.57.6:

| Format | Prompt tokens | Prefill ms | Decode ms/token | Decode tokens/s |
|---|---:|---:|---:|---:|
| FP32 | 59 | 138.96 | 88.18 | 11.34 |
| FP16 | 59 | 319.95 | 30.47 | 32.82 |
| Dynamic INT8 | 59 | 237.86 | 48.33 | 20.69 |
| FP32 | 990 | 1694.11 | 108.73 | 9.20 |
| FP16 | 990 | 5487.15 | 53.87 | 18.56 |
| Dynamic INT8 | 990 | 3592.63 | 60.88 | 16.43 |

Both lower-precision configurations had faster decode but slower prefill in
these observations. Fixed execution order and one sample per case do not
establish stable speedups; kernel-level causes were not profiled.

Quality uses nine prompts frozen before observing answers: three arithmetic,
three factual, and three instruction tasks. These use natural EOS and a
64-token cap, not forced-length speed outputs. The artifact retains prompts,
expected answers, generated text/IDs, stopping reasons, truncation, and scores.

| Format | Arithmetic tasks | Fact tasks | Instruction tasks | Total | Truncated |
|---|---:|---:|---:|---:|---:|
| FP32 | 3/3 | 3/3 | 2/3 | 8/9 | 0 |
| FP16 | 3/3 | 3/3 | 2/3 | 8/9 | 0 |
| Dynamic INT8 | 0/3 | 1/3 | 0/3 | 1/9 | 1 |

These are instruction-constrained task-success scores, not standalone arithmetic
or factual accuracy. INT8 gave the correct arithmetic results but added forbidden
explanations; its water/Mars answers also contained correct facts with extra text.
FP32 and FP16 returned identical answers on all nine tasks. Every format failed
the JSON task by adding Markdown fences. INT8 additionally failed word reversal
and produced an unrelated, truncated response to the lowercase task.

Dynamic INT8 converted all 169 linear layers, including `lm_head`. It quantizes
activations inside those operations, with FP32 inputs/outputs; embeddings, norms,
and KV caches remain FP32. FP16 parameters and observed KV caches are FP16.
`model.dtype` alone does not describe packed quantized weights. This is not
weight-only quantization or a model-memory measurement.

QNNPACK emitted its warning that `reduce_range=True` is ignored; it was not
suppressed. The experiment does not isolate that warning's effect, establish
general INT8 quality loss, or represent an MLX/GGUF backend. Nine simple tasks
are a sanity check, not a general quality benchmark.
See [the complete evidence and verification](logs/practice.md#cpu-precision-comparison-2026-11-06).

## Continuous batching with vLLM

`modal_continuous_batching.py` starts a private vLLM server and runs the
`benchmarks/continuous_batching.py` client workload inside one bounded T4 function.
The client controls arrivals; vLLM owns scheduling, model execution, and paged
KV-cache management. There is no custom engine scheduler, public endpoint,
persistent deployment, or change to the existing CPU API.

Recorded command:

```bash
~/.venvs/modal/bin/modal run modal_continuous_batching.py \
  --output benchmark-results/continuous-batching/t4-vllm-001.json
```

The file exists. Later runs require a fresh path and separate GPU approval.
The runner allows one container, no automatic retries, and a 600-second function
timeout. It terminates the server process group after collecting results.
Image-build failures happen before the local entrypoint can write its artifact;
the three encountered failures are preserved separately under
`benchmark-results/continuous-batching/t4-vllm-build-001.json` through `003.json`.

### Controlled configuration

- vLLM 0.33.0 in an official Linux/amd64 image pinned by digest.
- Same Qwen model/tokenizer revision as earlier experiments; Tesla T4, FP16
  weights and KV cache. Observed attention backend: `TRITON_ATTN`.
- Context limit 2048, at most two active sequences, 256-token scheduler budget,
  GPU memory utilization setting 0.5.
- Prefix caching disabled; chunked prefill enabled; eager execution.
  `torch.compile`, CUDA graphs, and engine JIT-kernel warm-up are disabled.
- The separate asynchronous-scheduling optimization is disabled. This does
  **not** disable continuous batching.
- Engine generation defaults are explicit: greedy requests, model EOS IDs
  supplied as stop-token IDs, and `min_tokens=max_tokens` for fixed-length cases.
  Natural-EOS control omits the minimum.
- Weights are downloaded during CPU image build and used offline during the
  GPU job. Local dependencies and earlier Torch environments are unchanged.

Both measured scenarios use the same engine, not vLLM versus Transformers:

1. One discarded 16-token warm-up request.
2. **Sequential:** A, then B, then C.
3. **Refill:** start A/B; after A finishes, submit C while B remains active.
4. One separate natural-EOS control.

A/B/C share the existing 59-token prompt and generate exactly 16/128/32 tokens.
HTTP/SSE is measured over container loopback, excluding build, loading, startup,
and warm-up. The artifact retains raw text/IDs, prompt-ID comparisons, usage,
request IDs, planned/actual timelines, chunk gaps, and acceptance checks.

### Observed refill and performance

All eight requests succeeded, their returned prompt IDs matched local formatting,
and every fixed-length request met its cap. The refill timeline, relative to
that scenario's start:

| Event | Time ms |
|---|---:|
| A first text | 31.99 |
| B first text | 736.16 |
| A `[DONE]` | 1851.91 |
| C request start | 1852.44 |
| C first text | 1899.32 |
| C `[DONE]` | 2625.34 |
| B `[DONE]` | 4525.47 |

C produced text before B finished: waiting for the entire original group was
not required. These are client-observed streaming events, not an engine
scheduler or GPU-kernel trace.

| Scenario | Output tokens | Elapsed ms | Aggregate output tokens/s |
|---|---:|---:|---:|
| Sequential | 176 | 4172.22 | 42.18 |
| Refill | 176 | 4525.87 | 38.89 |

Refill throughput was **7.81% lower in this observation**, despite successful
admission and overlap. One batch-1 warm-up does not establish that two-active-
request kernel paths are warm. Fixed order, single samples, eager settings, and
different arrival schedules limit interpretation; no cause was isolated by
profiling. This is not a static-batching ablation or a production speed claim.

The natural control stopped at EOS after 10 tokens, below its cap of 32.
It returned `Hello! How can I assist you today?`, so it passed stopping/stream
checks but did not obey the one-word instruction. A/C token IDs matched between
measured scenarios; B differed starting at token 92. Output equality and
technical correctness are not established by successful serving.

The full suite passed 145 tests, and the real GPU job passed all acceptance
checks. See [the detailed evidence](logs/practice.md#t4-continuous-batching-with-vllm-2026-11-06)
for request latencies, deployment fixes, warnings, and limitations.

## Transformers versus vLLM comparison

`modal_engine_comparison.py` runs the existing Transformers HTTP service, then
vLLM, sequentially inside one T4 allocation. The baseline worker exits before the
engine starts, so the two models do not compete for GPU memory. Both use the
same pinned vLLM image, Qwen checkpoint, tokenizer, FP16 precision, and HTTP client.
This is the existing application on the comparison's cloud software stack,
not a reproduction of the older local package environment.

The new `benchmarks/engine_comparison.py` injects a benchmark-only runtime into
the real `src.api.app`. Production loading, one-worker scheduling, the two-slot
queue, cancellation, and streaming remain unchanged. The normal API still loads
CPU FP32. Its strict request schema is not expanded: only vLLM receives its
vendor-specific streaming/token/minimum-generation fields.

### Protocol and bounds

Each backend receives 14 requests:

| Phase | Requests | Used for comparison |
|---|---:|---|
| Sequential warm-up | A/B/C | No |
| Concurrent/refill warm-up | A/B/C | No |
| Sequential measurement | A/B/C | Yes |
| Concurrent/refill measurement | A/B/C | Yes |
| Natural-EOS warm-up | Hello | No |
| Natural-EOS control | Hello | Stopping control only |

A/B/C use the shared short prompt and exact output lengths 16/128/32, producing
176 completion tokens per fixed scenario. Sequential admission waits for each
predecessor. Refill starts A/B and admits C after A completes, with at most two
HTTP requests in flight. Both backends receive the same closed-loop policy,
not an identical externally timed arrival trace.

Transformers suppresses EOS through a scoped change to each fresh generation
configuration; vLLM receives `min_tokens=max_tokens`. The baseline scope restores
the setting on success and failure, and never mutates shared model defaults.
The natural controls omit the minimum and have a 32-token cap.

Startup and discarded warm-ups are excluded from scenario throughput. Results
retain individual first-text/E2E times, usage, output text, request timelines,
and actual token IDs where available. Baseline input-ID matching is observed
at the model boundary during discarded warm-ups, not invented HTTP fields.
Baseline non-overlap is expected and is not an acceptance failure.

One observation per scenario is not a stable speedup claim. Chunk delivery gaps
are not token/kernel timings. Fixed execution order, engine kernels, stream
buffering, and different actual completion-driven arrivals remain confounders.
No percentiles, quality equivalence, or memory-saving claim is inferred.

The cloud job allows one T4 container, four CPUs, 8192 MiB host memory, zero
minimum containers, and no retries. A shared 480-second internal deadline spans
both backends within a 600-second hard function limit. Servers remain on loopback;
there is no persistent endpoint. Process-group cleanup checks surviving
descendants even after the original leader exits and records forced termination.
Baseline failures abort before starting vLLM.

### Commands and verification

The entrypoint works in the separate SDK-only environment:

```bash
~/.venvs/modal/bin/modal run modal_engine_comparison.py --help
```

Cloud execution requires explicit approval and a fresh output path:

```bash
~/.venvs/modal/bin/modal run modal_engine_comparison.py \
  --output benchmark-results/engine-comparison/t4-transformers-vllm-001.json
```

Local verification passed **153 tests** and a real cached-Qwen, CPU-FP32
loopback HTTP exercise of all 14 requests. Every fixed count matched, both
natural controls stopped after 10 tokens, warm-up input IDs matched, model
defaults were preserved, and server resources were released. That artifact,
`benchmark-results/engine-comparison/cpu-http-smoke-001.json`, is functional
evidence, not a GPU performance baseline.

Additional real-process smoke checks covered ordinary termination, forced
termination, and a surviving descendant after leader exit. Controlled failure
checks preserved partial child results and remote errors, rejected failed
baselines, and refused to overwrite existing evidence.

### Recorded T4 comparison

The command above completed one approved GPU invocation and saved the named
artifact. That path now exists and must not be reused. All 28 requests returned
HTTP 200 with distinct request IDs; all cross-backend acceptance checks passed.

| Backend | Policy | Output tokens | Scenario ms | Output tokens/s |
|---|---|---:|---:|---:|
| Transformers | Sequential | 176 | 4074.64 | 43.19 |
| vLLM | Sequential | 176 | 4143.77 | 42.47 |
| Transformers | Concurrent/refill | 176 | 4033.85 | 43.63 |
| vLLM | Concurrent/refill | 176 | 2983.86 | 58.98 |

In this observation, vLLM throughput was **35.19% higher under the refill
policy**, but **1.67% lower under sequential admission**, compared with
Transformers. This is a whole-serving-stack comparison, not an isolation of
batching or a particular attention kernel.

For C under refill, first-text latency from its own request start was
2987.20 ms on Transformers versus 46.77 ms on vLLM. On vLLM, C produced text
at scenario time 420.20 ms and finished at 1139.15 ms, before B finished at
2983.27 ms. The baseline kept C behind B, as its unchanged scheduler requires.

First-text boundaries also differ in buffering: the first sequential A content
chunk was `An ` on Transformers and `An` on vLLM. These are user-visible text
measurements, not comparable internal first-token timestamps.

Sequential A/B/C text matched across backends; refill A/C matched but B differed.
Actual baseline output IDs were not captured, so no cross-backend token-ID
parity is claimed. Both natural controls returned the same 10-token answer and
stopped early, while failing the one-word instruction. Generated B explanations
of prefill remained incorrect; this experiment does not establish output quality.

Both process groups exited with code zero, without outer-runner SIGKILL
escalation. vLLM's own shutdown still logged force-killing EngineCore and a
semaphore cleanup warning. Modal reported the app completed; no endpoint remains.
See the [full comparison evidence](logs/practice.md#transformers-versus-vllm-comparison-2026-11-06)
for per-request measurements, runtime versions, and limitations.

## Explicit two-model routing

`src/model_router.py` is an opt-in FastAPI app that owns two pinned checkpoints
behind one private, single-resident vLLM process group from
`src/vllm_engine.py`. It runs only on the pinned Linux/T4 image; the production
CPU API is unchanged.

### Policy and bounds

- The request `model` field must be exactly `Qwen/Qwen2.5-0.5B-Instruct` or
  `Qwen/Qwen3-1.7B`. Unknown identifiers return 422 before any load or process
  work. No request value reaches a path, download, or command.
- One model is resident at a time. At most two completions are active and two
  more wait in a strict FIFO queue shared across models, so a later low-model
  request cannot overtake a queued high-model request.
- A model switch drains active work, stops the process group, and loads the next
  checkpoint. One lifecycle pump owns every load and unload, so simultaneous
  requests cannot trigger duplicate loads.
- Each request is pinned before admission: formatted-prompt token IDs, chat
  template hash, EOS IDs, and a 2048-token context bound are computed with the
  same revision's tokenizer. Prompt plus `max_tokens` over that bound returns
  400 `context_length_exceeded` without loading a model.
- Responses are validated, not trusted: upstream `id` must equal
  `chatcmpl-<request-id>`, served model must match, prompt token IDs must match
  the pinned preparation, completion token IDs are forwarded, usage is checked
  against the prompt and output cap, and thinking/reasoning fields are rejected.
  JSON and SSE share these checks; a truncated or malformed stream returns an
  error event and no success marker.
- A failed load, lost process, or failed upstream closes admission and fails
  queued work with a sanitized error instead of retrying or switching models.
  Shutdown drains, stops the group, and reports unclean cleanup.

### Commands and verification

Local behavioral tests use a fake engine plus a real loopback ASGI upstream:

```bash
.venv/bin/python -m unittest discover -s tests
```

That suite passed **182 tests**, including 24 router and 5 operations tests covering FIFO order,
capacity, cold-load sharing, drain before switch, queued and active cancellation,
load/process/cleanup failures, strict schema rejection, identity mismatches,
missing prompt IDs, truncated streams, and shutdown ownership.

Cloud execution uses two separately approved, gated invocations; `--help` needs
neither the Modal SDK nor a GPU:

```bash
~/.venvs/modal/bin/modal run modal_model_routing.py \
  --mode smoke --output benchmark-results/model-routing/t4-qwen3-smoke-001.json
~/.venvs/modal/bin/modal run modal_model_routing.py \
  --mode lifecycle --output benchmark-results/model-routing/t4-two-model-lifecycle-001.json
```

### Recorded Qwen3 smoke (2026-11-08 UTC)

One T4 invocation, three requests, one model load, zero retries. All 15 checks
passed; the artifact is
`benchmark-results/model-routing/t4-qwen3-smoke-001.json`.

| Case | Stream | Prompt tokens | Output tokens | Finish | First text ms | E2E ms | Output |
|---|---|---:|---:|---|---:|---:|---|
| Reply with one word | No | 19 | 2 | `stop` | - | 77787.31 | `Hello` |
| 2 + 2, integer only | Yes | 26 | 2 | `stop` | 70.38 | 100.55 | `4` |
| Count to 200, cap 1 | Yes | 35 | 1 | `length` | 38.53 | 39.54 | `1` |

The first E2E time includes the cold load of Qwen3-1.7B (75076.42 ms to ready).
Device-level `nvidia-smi` used memory was 4 MiB before the load, 7606 MiB with
Qwen3 resident, and 4 MiB after cleanup; the process group stopped in 708.11 ms
with `forced: false`, no error, and exit code zero. vLLM still logged
force-killing its EngineCore and one leaked-semaphore warning during its own
shutdown. These warnings were not suppressed.

### Recorded two-model lifecycle (2026-11-08 UTC)

One approved T4 invocation, 26 completion attempts, three model loads, zero
retries. All 24 checks passed; the artifact is
`benchmark-results/model-routing/t4-two-model-lifecycle-001.json`.

| Order | Resident model | Startup ms | Unload ms | Forced kill |
|---|---|---:|---:|---|
| 1 | `Qwen/Qwen2.5-0.5B-Instruct` | 73560.54 | 605.72 | No |
| 2 | `Qwen/Qwen3-1.7B` | 65881.53 | 505.42 | No |
| 3 | `Qwen/Qwen2.5-0.5B-Instruct` | 48008.74 | 556.30 | No |

Observed lifecycle behavior:

- Two warm-up and nine frozen quality requests ran on the low model before any
  switch; eight of nine quality tasks passed.
- One low-model request stayed active while a high-model request waited. The
  high model loaded only after the active lease was released, and the later
  low-model request behind it was not admitted first.
- A third low-model request was canceled while queued: it produced a `canceled`
  manager event and no `admitted` event, and the HTTP client observed no
  completion.
- Two high-model requests were queued during the switch and were admitted
  together afterwards. Both hit their 128-token cap with exactly 128 completion
  tokens and `finish_reason: length`.
- Returning to the low model triggered the third load; the unknown-model request
  returned 422 without another load.
- Frozen quality scores: low model 8/9, high model 9/9. Only the low model's JSON
  task failed, with the answer wrapped in a ```json fence. Single greedy samples
  of nine tasks are not a quality ranking.

Device-level `nvidia-smi` memory was 6 MiB before the first load, about 7.6 GiB
while either model was resident, and 6 MiB after cleanup. This metric cannot
separate weights from vLLM's KV-cache reservation, because both models run with
`--gpu-memory-utilization 0.5` on a 15360 MiB T4. After a warm prompt, first
text was about 31.5 ms on the low model and 37.9-47.8 ms on the high model, but
requests issued during a switch included 48-74 s of drain, load, and admission.
Those are not matched latency observations.

vLLM again logged force-killing its EngineCore and one leaked-semaphore warning
during shutdown; the outer router still reported every unload as stopped,
unforced, error-free, and exit code zero. All 25 completed requests had distinct
request IDs and `chatcmpl-<request-id>` response identities, and every returned
prompt tokenization matched the router's own preparation.

### Limits

These are single observations on one T4. They do not establish throughput,
relative model quality, GPU kernel overlap, or a stable latency comparison.
Timing includes queueing and model switches where noted; memory is whole-device
`nvidia-smi` usage, not peak allocation. Failure injection (crashed engine,
failed load, unclean cleanup) is covered only by deterministic local tests, not
by a GPU run. See the
[practice log](logs/practice.md#explicit-two-model-routing-2026-11-08-utc)
for the full evidence and limits.
