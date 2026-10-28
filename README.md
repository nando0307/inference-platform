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
The suite has 110 passing tests, with real-Qwen lifecycle and HTTP smoke evidence.
See the [Phase 4 evidence](logs/practice.md#phase-4-bounded-concurrency-and-cancellation)
for the fixed-workload comparison and its limitations. Next: explain queue wait,
latency, throughput, and cancellation ownership before introducing batching.
Independent learning checks remain pending; automated verification does not
prove understanding.
The current GPU exploration has a completed five-sample warm T4 run:
3397.13 ms median inference duration and 36.86 output tokens/s for the 59/128 case.

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
These differ from the recorded local benchmark stack; differences cannot be
attributed solely to hardware. Transitive dependencies are not fully locked.

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

## Run the local application

From the repository root, using the existing project environment:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

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
