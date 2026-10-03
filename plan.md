# Mini Inference Platform: Project and Learning Plan

## 1. Project Goal

Build a small inference platform while learning how language models generate
text, how inference services handle requests, and how performance changes under
different workloads.

The final system should be able to:

- Load and run a language model.
- Accept requests through an API.
- Stream generated text.
- Measure latency, throughput, and memory.
- Handle concurrent requests with bounded resource usage.
- Process requests in batches.
- Explain and measure KV-cache behavior.
- Compare precision and quantization choices.
- Use a production inference engine.
- Select between multiple models.
- Expose operational metrics and run in a reproducible deployment.

The goal is not to reproduce every feature of a commercial platform.

The goal is to understand the important mechanisms and demonstrate their behavior
with evidence.

---

## 2. Working Agreement

### My responsibilities as the learner

- Write and run the code myself.
- Read the relevant documentation.
- Predict what an experiment will show before running it.
- Inspect intermediate values instead of treating libraries as black boxes.
- Keep working checkpoints in Git.
- Record unexpected results and investigate them.
- Explain why my implementation works.

### My mentor's responsibilities

- Give bounded assignments and measurable acceptance criteria.
- Explain concepts and point to relevant documentation.
- Review my code and reasoning.
- Help distinguish application bugs, environment problems, and library issues.
- Offer hints before complete solutions.
- Avoid modifying my files unless I explicitly request it.

### How I will ask for help

For conceptual questions:

    What I am trying to understand:
    My current explanation:
    What does not make sense:

For debugging:

    What I expected:
    What happened:
    Relevant code:
    Exact command:
    Complete error or unexpected output:
    What I already checked:

For experiments:

    Hypothesis:
    Variable changed:
    Variables held constant:
    Results:
    My interpretation:
    Remaining uncertainty:

### Keep the roadmap separate from the lab notes

Use `plan.md` as the stable roadmap. Keep the active assignment, predictions,
commands, results, and open questions in a separate lab notebook such as
`docs/lab-notes.md` when the README becomes crowded. Do not create empty
documentation files in advance.

Work on one bounded question at a time. End each assignment with a short
"what I learned" note and a working Git checkpoint. A stage is complete when
its acceptance criteria are demonstrated, not when every related topic has
been explored. Record optional questions rather than expanding the current task.

---

## 3. Current Baseline

### Environment

- Apple Silicon Mac with 16 GiB of memory.
- Project-local Python 3.12 virtual environment.
- PyTorch 2.14.1 and Transformers 5.18.0 were successfully imported.
- Initial model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Initial execution target: CPU with float32 weights.

### Already demonstrated

- [x] Install and import PyTorch and Transformers.
- [x] Download and load the model.
- [x] Load the matching tokenizer.
- [x] Accept a terminal prompt.
- [x] Generate and print a response.

### Still to establish

- [ ] Pin the working dependency versions.
- [ ] Record the model revision for reproducible experiments.
- [ ] Make generation settings explicit.
- [ ] Inspect logits and token selection.
- [ ] Understand and reproduce a small generation loop.
- [ ] Document observations rather than only installation commands.

A successful response proves that the execution path works. It does not prove
that the response is factually correct.

### Revised sequence

1. Single-model inference: tokenization, logits, uncached and cached loops,
   sampling, then a controlled CPU/MPS comparison.
2. A local API: a small chat-completions contract, serialized inference, and
   streaming.
3. Benchmarks: token timing, throughput, memory, and profiling.
4. Controlled concurrency: a bounded queue, cancellation, and backpressure.
5. An explicit hardware decision, then static batching and microbatching.
6. A deeper KV-cache memory study.
7. Precision and quantization experiments on supported backends.
8. One inference engine, using the same benchmark client.
9. Explicit multi-model routing.
10. Monitoring and deployment.

Basic cache mechanics move into Stage 1 so that prefill and decode measurements
in Stage 3 have an explanation. Stage 6 retains the deeper memory investigation.

---

## 4. Engineering Rules

1. Keep a working baseline before adding a feature.
2. Change one experimental variable at a time.
3. Prefer explicit behavior over hidden defaults.
4. Do not add a framework or abstraction without a concrete need.
5. Load the model once per serving process, not once per request.
6. Keep environments, model weights, credentials, and generated logs out of Git.
7. Save files before running or staging them.
8. Review staged changes before committing.
9. Do not disable diagnostics merely to hide unexplained problems.
10. Do not expose the service publicly before adding resource limits and access
controls.

### What belongs in Git

- Source code.
- Dependency specifications.
- Small configuration files without secrets.
- Documentation.
- Small benchmark summaries.
- Tests that protect meaningful behavior.

### What does not belong in Git

- `.venv/`
- Model weights.
- Hugging Face credentials.
- Private prompts or user conversations.
- Large generated benchmark artifacts.
- Cache directories.

---

# Stage 1: Understand Single-Model Inference

## Central question

How does an input sequence become a generated response?

## Mental model

    Chat messages
        ↓
    Chat template
        ↓
    Tokenizer
        ↓
    Token IDs
        ↓
    Embeddings → transformer layers → language-model head
        ↓
    Next-token logits
        ↓
    Greedy selection or sampling
        ↓
    One new token ID
        ↓
    Append and repeat
        ↓
    Stop at EOS or the generation limit
        ↓
    Decode generated IDs into text

## Concepts

- Tokens and token IDs.
- Tokenizer vocabulary.
- Chat templates and special tokens.
- Input tensors and attention masks.
- Model weights and numerical precision.
- Logits and softmax.
- Greedy decoding and sampling.
- Temperature, top-k, and top-p.
- End-of-sequence tokens.
- `max_new_tokens`.
- Evaluation mode and inference mode.
- An introductory distinction between prefill and decode.
- Tied input embeddings and output-head weights: inspect
  `model.config.tie_word_embeddings` rather than assuming separate matrices.

## Assignments

### 1.1 Inspect tokenization

- [ ] Print token pieces and IDs for several short inputs.
- [ ] Compare capitalization, punctuation, and whitespace.
- [ ] Decode the IDs back into text.
- [ ] Compare plain-text tokenization with chat-formatted tokenization.
- [ ] Explain why the tokenizer must match the model.

### 1.2 Inspect one next-token prediction

- [ ] Run one model forward pass without calling `generate()`.
- [ ] Print the logits shape.
- [ ] Select the logits for the next token after the input.
- [ ] Convert them to probabilities.
- [ ] Display the five highest-probability token candidates.
- [ ] Explain which candidate greedy decoding would select.

### 1.3 Build a short manual greedy loop

- [ ] Start without a KV cache.
- [ ] Select one next token.
- [ ] Append it to the input IDs.
- [ ] Update the attention mask.
- [ ] Repeat for a small number of tokens.
- [ ] Stop on the model's configured EOS tokens.
- [ ] Decode only the newly generated IDs.

First compare with `generate()` using `use_cache=False` on both paths.
Match the exact input IDs, model revision, device, precision, attention mask,
stopping rules, and token limit. Set `do_sample=False`, `num_beams=1`, and
`repetition_penalty=1.0` without other token constraints.

Read stopping IDs from `model.generation_config.eos_token_id`, which can be
a list. The selected Qwen checkpoint's generation configuration lists
`[151645, 151643]`; do not check only one ID or hardcode these for other models.

If token sequences differ, locate the first differing token and compare the
logits on the same prefix. Check implementation and configuration differences
before attributing anything to numerical precision. A near tie is an explanation
only when the measured logits support it, not a reason to dismiss a mismatch.

### 1.4 Add basic KV-cache reuse

- [ ] Extend the working manual loop to reuse `past_key_values`.
- [ ] Process the prompt once during prefill.
- [ ] Pass only unprocessed tokens on subsequent decode steps.
- [ ] Maintain the full attention mask and any cache-position inputs required
  by the installed model API.
- [ ] Compare with the uncached loop and with cached `generate()` using matched
  decoding settings.
- [ ] Explain why the first-token work differs from later-token work.

Keep this experiment short and single-request. Leave cache allocation
strategies, memory scaling, and paged attention for Stage 6.

### 1.5 Explore sampling

- [ ] Compare greedy decoding with sampling.
- [ ] Change temperature while holding other controls fixed.
- [ ] Change top-k with top-p filtering disabled.
- [ ] Change top-p with top-k filtering disabled.
- [ ] Repeat experiments with documented seeds.
- [ ] Explain why a higher temperature does not imply greater factual accuracy.

Print and record the effective generation configuration. Qwen's saved defaults
include sampling enabled, temperature `0.7`, top-k `20`, top-p `0.8`, and
repetition penalty `1.1`; leaving an argument unset does not disable it.

Use this explicit sampling baseline before varying one control:

    do_sample=True
    num_beams=1
    temperature=1.0
    top_k=0
    top_p=1.0
    repetition_penalty=1.0

For a temperature-only experiment, keep top-k disabled and top-p at `1.0`.
For a top-k experiment, change top-k while leaving top-p at `1.0`.
For a top-p experiment, change top-p while leaving top-k at `0`.
Keep the same small token limit and record the seed for each run.

Use `do_sample=False` for the separate greedy baseline; do not pass a zero
temperature to Transformers as a substitute. Seeds aid comparisons within a
fixed environment but do not guarantee identical output across backends.

### 1.6 Compare execution devices

- [ ] Preserve the working CPU float32 result.
- [ ] Check `torch.backends.mps.is_available()`.
- [ ] Run the same small workload on MPS float32 if supported.
- [ ] Keep model weights, inputs, and generation settings otherwise unchanged.
- [ ] Record output differences and observed performance without assuming the
  GPU must win for a small model.

Do not change device and precision in the same first comparison. Stage 7 adds
lower-precision runs after Stage 3 establishes reliable timing.

## Acceptance criteria

- I can explain every transformation from messages to output text.
- I can interpret the dimensions of the logits tensor.
- I can distinguish logits from probabilities.
- I can explain why generated sequences include the input prefix.
- I can explain why generation may stop before `max_new_tokens`.
- I can compare my manual loop with the library implementation.
- I can explain prefill and decode using my cached and uncached loops.
- I distinguish a device change from a precision or generation-setting change.

## Evidence

Lab notes containing actual tokenization output, next-token candidates,
manual-loop comparisons, sampling settings, and CPU/MPS observations.

## Suggested commit

`Add next-token inspection and generation experiments`

---

# Stage 2: Expose a Local API

## Central question

How does a script become a long-running inference service?

## Concepts

- HTTP requests and responses.
- Request validation.
- Application startup and shutdown.
- Model lifecycle.
- Error responses.
- Streaming.
- Readiness versus process liveness.

## Assignments

- [ ] Separate reusable inference logic from terminal input/output.
- [ ] Load the tokenizer and model during application startup.
- [ ] Implement a documented subset of `POST /v1/chat/completions`.
- [ ] Accept a fixed supported `model`, `messages`, `max_tokens`, `temperature`,
  `top_p`, and `stream`; reject unsupported features rather than ignoring them.
- [ ] Return generated text and token counts.
- [ ] Bound input and output tokens together against the supported context
  window, including chat-template tokens; reject excess input rather than
  silently truncating it.
- [ ] Reject invalid parameter combinations clearly.
- [ ] Add a readiness endpoint that reports whether inference is available.
- [ ] Add `stream=true` server-sent events after ordinary responses work.
- [ ] Keep the service bound to localhost initially.
- [ ] Serialize model execution off the event loop; while busy, reject new work
  with a documented response instead of building an unbounded wait queue.
- [ ] Add basic request IDs, completion/error status, and duration logging
  without recording private prompts.

Do not begin with a complicated backend interface. Extract only what the API
actually needs.

### Minimal API contract

Use `max_tokens` as the external output cap and map it to Transformers'
`max_new_tokens`. Define API `temperature=0` as greedy decoding; map positive
temperature to sampling instead of passing zero to Transformers. Specify
remaining internal defaults, including top-k and repetition penalty.

Return a model identifier, assistant message, finish reason, and token usage.
For streaming, document `choices[].delta.content`, the terminal finish reason,
and the `[DONE]` event. This is a small compatible subset, not a promise to
implement every OpenAI API feature. Tools, images, and reasoning-specific fields
are outside this stage.

### Safe execution from the first endpoint

A synchronous FastAPI endpoint may execute in a threadpool, allowing several
requests to enter the model concurrently. An asynchronous endpoint that calls
blocking generation directly can block the event loop. Neither gives the
serialized baseline automatically.

Use one controlled background execution path and a busy/admission guard.
Hold ownership until generation has actually ended, including on errors or
client disconnects. Stage 4 replaces immediate busy rejection with a bounded
queue and adds detailed cancellation behavior.

### Streaming text correctly

Investigate `TextIteratorStreamer` for incremental detokenization and a
background producer. A token is not necessarily a complete word or Unicode
character, and a text chunk is not necessarily one token. Do not independently
decode each token and blindly concatenate the strings.

Ensure failures and disconnects terminate the producer/consumer flow rather
than leaving the reader waiting forever. Streamer text output is suitable for
measuring visible delivery, but not by itself for exact first-token timing.

## Acceptance criteria

- Multiple requests reuse the already-loaded model.
- Invalid requests receive useful errors.
- The server does not claim readiness before model loading succeeds.
- Ordinary and streaming responses have a documented contract.
- Streaming begins before the complete response has been generated.
- Two simultaneous requests cannot accidentally start two generation calls.
- Requests beyond the supported context window are rejected clearly.
- A producer failure terminates the stream and releases inference ownership.

## Evidence

Documented commands that exercise:

- A successful request.
- An invalid request.
- An oversized request.
- A streaming request.
- Two overlapping requests demonstrating the initial busy response.
- A streaming disconnect or generation failure.

## Suggested commit

`Expose local inference through an API`

---

# Stage 3: Build a Reproducible Benchmark

## Central question

Where does time go, and what limits performance?

## Concepts

- Model download time.
- Startup and model-loading time.
- Warm-up.
- End-to-end latency.
- Time to first token.
- Time to first delivered text.
- Time per output token (TPOT) and inter-token latency.
- Aggregate throughput.
- Memory consumption.
- Median and tail latency.

## Assignments

- [ ] Create a small fixed collection of benchmark prompts.
- [ ] Include short and longer inputs.
- [ ] Implement the load client as a separate program that can also target the
  Stage 8 engine by changing its URL and declared configuration.
- [ ] Measure prompt lengths after chat formatting in tokens, not characters;
  distinguish actual input tokens from batch-padding positions.
- [ ] Record actual input and generated token counts.
- [ ] Measure startup separately from request processing.
- [ ] Warm up the model before steady-state measurements.
- [ ] Run repeated requests rather than reporting one result.
- [ ] Report sample counts, median latency, and p95 latency.
- [ ] Measure memory using a clearly named metric.
- [ ] Record hardware, software versions, model revision, and generation
settings.
- [ ] Instrument token production separately from streamed-text delivery.
- [ ] Profile a representative slow case with `torch.profiler` or `py-spy`
  instead of guessing the bottleneck from total duration.
- [ ] Keep profiling runs separate from headline performance measurements
  because instrumentation can alter timing.

### Define the metrics before collecting them

- **End-to-end latency:** client request start to completed response.
- **Model-side TTFT:** inference start to production of the first generated
token.
- **Mean TPOT:** `(last_token_time - first_token_time) / (N - 1)` for `N > 1`,
  using generated-token timestamps from the same clock. Report it as
  unavailable for shorter responses, not zero.
- **Inter-token latency:** the interval between adjacent generated-token
  timestamps; report its distribution separately from mean TPOT.
- **Client first-output latency:** request start to receipt of the first
non-empty streamed output.
- **Aggregate output throughput:** total generated tokens divided by the
benchmark's elapsed time.
- **Queue wait:** time between admission and inference execution, once a queue
exists.

Report the request count and error/cancellation count with each latency
distribution. Distinguish successful useful throughput from work spent on
failed or canceled requests, and state whether EOS tokens are counted.

### Instrumentation and workload modes

`generate()` returns after completion, so timing only the call yields total
generation time, not TTFT. Use the manual loop or a token-ID callback/streamer
that timestamps actual newly generated tokens. Exclude the initial prompt from
these callbacks. Use one clock per duration; do not subtract a server timestamp
from a client timestamp on another machine.

`TextIteratorStreamer` can buffer until text is ready to display. Its first
non-empty text chunk measures visible output latency, not necessarily the first
model token. Keep both measurements named accurately.

Run two explicitly labeled workloads:

- **Natural stopping:** preserve EOS behavior, record actual output counts,
  and measure realistic serving latency and quality.
- **Fixed-length throughput:** set matching `min_new_tokens` and
  `max_new_tokens` for a small synthetic run, suppressing ordinary EOS stopping
  until the chosen length. Document other stopping conditions; do not override
  cancellation or resource limits. This workload is not a quality evaluation.

Use short and longer prompts with separate output-length controls to distinguish
prefill effects from decode effects. Repeat enough runs to expose variability;
do not treat a small-sample p95 as a stable production estimate.

A streamed text chunk is not necessarily one token.

For accelerator measurements, account for asynchronous execution when timing
isolated operations. Do not insert global synchronization into every production
request.

On Apple Silicon, distinguish process memory from device allocations and
unified-memory measurements.

## Acceptance criteria

- Download and startup are not mixed into steady-state generation results.
- Every result includes actual generated token counts.
- The same experiment can be repeated.
- Latency and throughput are reported separately.
- Small sample sizes are acknowledged when discussing p95.

## Evidence

A repeatable results table, a representative profile, and an evidence-backed
explanation of the main bottleneck, including measurement limitations.

## Suggested commit

`Add repeatable inference benchmarks`

---

# Stage 4: Add Controlled Concurrency

## Central question

What happens when requests arrive faster than inference can finish them?

## Concepts

- Blocking work versus asynchronous I/O.
- Queueing.
- Backpressure.
- Admission control.
- Cancellation.
- Timeouts.
- Shared model state.
- Per-process model memory.

## Initial design

    Incoming requests
        ↓
    Validation and admission
        ↓
    Bounded queue
        ↓
    One inference worker
        ↓
    Response or streamed output

Extend Stage 2's single-worker baseline with a bounded queue. Do not introduce
parallel model calls merely because several HTTP clients are connected.

## Assignments

- [ ] Keep the event loop responsive while inference runs.
- [ ] Introduce a bounded request queue.
- [ ] Define behavior when the queue is full.
- [ ] Measure queue wait separately from generation time.
- [ ] Remove canceled requests that have not started.
- [ ] Define how running generation stops after cancellation.
- [ ] Handle shutdown without accepting work that cannot finish.
- [ ] Test with multiple simultaneous clients.
- [ ] Define request transitions: queued, running, completed, failed, or canceled.
- [ ] Preserve per-request settings and document how seeded sampling behaves
  without accidentally sharing or resetting random state across requests.

An `async def` endpoint does not automatically make model computation
non-blocking.

Adding server processes may load multiple copies of the model.

Use a cooperative cancellation flag and a Transformers `StoppingCriteria` to
stop a running generation between steps. Canceling the HTTP coroutine or a
thread's future does not itself stop computation. Measure cancellation delay
and retain the worker slot until the generation call exits.

Test queued cancellation separately from running cancellation. Use bounded
stream buffers and define what happens when a client consumes output too slowly.

## Acceptance criteria

- Overload does not create an unbounded queue.
- One client's text never appears in another client's response.
- Health/readiness behavior remains understandable under load.
- Cancellation has an observable effect.
- I can explain why throughput and latency change with concurrency.

## Evidence

Compare concurrency levels such as 1, 2, and 4 while holding the workload
constant.

## Suggested commit

`Add bounded request scheduling and cancellation`

---

# Stage 5: Introduce Batching

## Central question

When does processing several requests together improve efficiency?

## Hardware decision before batching and engine comparisons

- [ ] Choose and record a target for Stages 5-8 before making GPU performance
  claims.
- [ ] Keep CPU as a correctness baseline and use MPS for local accelerator
  experiments when supported.
- [ ] Check current backend/model support for the intended inference engine.
- [ ] If choosing a rented Linux NVIDIA GPU, set a spending limit and shutdown
  procedure before provisioning it; no cloud spending is assumed or automatic.

CPU batching can still be useful, but neither CPU nor MPS guarantees a speedup
for this workload. Use measurements rather than declaring one platform faster
in advance. On the Mac, record which utilization and memory metrics are actually
available.

vLLM documents Apple Silicon CPU support and a separate vLLM-Metal GPU backend.
Do not assume vLLM and SGLang have identical hardware support or that CUDA
instructions apply to macOS.

The same client makes workloads consistent across hosts; it does not isolate
engine improvements if hardware also changes. For an engine comparison, run
both the Transformers baseline and the engine on the chosen target where
possible. Otherwise label the result a whole-system comparison.

## Concepts

- Batch dimension.
- Padding.
- Attention masks.
- Different prompt lengths.
- Different completion lengths.
- Static batching.
- Microbatching.
- Throughput versus per-request latency.

## Assignments

- [ ] Compare sequential processing with a small static batch.
- [ ] Use prompts of different lengths.
- [ ] Use the correct padding behavior for decoder-only generation.
- [ ] Preserve the mapping between requests and responses.
- [ ] Handle sequences that finish at different times.
- [ ] Measure aggregate throughput and memory.
- [ ] Only then experiment with a short batching wait window.

Do not implement a production continuous-batching scheduler yet.

## Acceptance criteria

- Each response belongs to the correct input.
- Padding does not contaminate the displayed response.
- Results compare equivalent amounts of work.
- I can explain when batching improves throughput but worsens latency.
- I do not assume batching must be faster on every device.

## Evidence

Compare batch sizes 1, 2, and 4 with documented lengths and settings.

## Suggested commit

`Add batched inference and compare performance`

---

# Stage 6: Investigate KV-Cache Memory and Scaling

## Central question

How does cached attention state scale with context, batch size, and allocation
strategy?

## Important context

Stage 1 already introduced prefill, decode, and a working cached loop.
This stage extends that understanding to memory growth and allocation behavior.
Keep a cache-disabled comparison available; do not assume the original
`generate()` implementation ran without caching.

## Concepts

- Prefill.
- Decode.
- Keys and values.
- Per-layer cache state.
- Cache length.
- Cache memory.
- Query heads versus KV heads.
- Dynamic and static cache strategies.
- Grouped-query attention (GQA) and its effect on cache size.
- A conceptual introduction to paged cache allocation before Stage 8.

## Assignments

- [ ] Compare generation with caching enabled and disabled.
- [ ] Keep model, prompt, output limits, precision, and decoding settings fixed.
- [ ] Inspect cache tensor shapes.
- [ ] Explain how cache length grows.
- [ ] Reuse the cached loop from Stage 1 rather than implementing it again.
- [ ] Keep attention masks and cache positions consistent with the library API.
- [ ] Compare short and longer prompts.
- [ ] Vary context length and batch size independently.
- [ ] Compare dynamic and static cache allocation where the backend supports it.
- [ ] Explain what paged allocation changes without building an allocator.

### Memory reasoning exercise

For a simple uniformly shaped cache, estimate:

    2
    × number of layers
    × batch size
    × cached sequence length
    × number of KV heads
    × head dimension
    × bytes per element

The factor of 2 represents keys and values.

Explain why the actual allocation may differ because of padding, preallocation,
metadata, or cache implementation.

For this Qwen checkpoint, inspect and verify these configuration fields:

- `num_hidden_layers`: 24.
- `num_key_value_heads`: 2.
- `num_attention_heads`: 14.
- `hidden_size`: 896.
- Head dimension for this architecture: `hidden_size / num_attention_heads`.

Calculate the per-token cache cost yourself for the actual cache dtype.
Compare it with a hypothetical model with the same dimensions but 14 KV heads.
That head-count change multiplies the ideal KV tensor storage by seven; it
does not imply a sevenfold change in total process memory or inference speed.

## Acceptance criteria

- I can explain prefill versus decode.
- I understand that a KV cache stores attention state, not completed answers.
- I can show the speed/memory tradeoff.
- My cached loop does not repeatedly submit the whole prefix as new uncached
input.
- I inspect output differences rather than assuming every backend produces
bit-identical results.

## Evidence

A context-length/batch-size memory study, an analytical cache-size estimate,
and an explanation of dynamic, static, and paged allocation tradeoffs.

## Suggested commit

`Add KV-cache experiments and memory analysis`

---

# Stage 7: Explore Precision and Quantization

## Central question

What changes when model weights use fewer bits?

## Concepts

- Float32, float16, and bfloat16.
- Weight quantization.
- Activation precision.
- KV-cache precision.
- Quantization error.
- Backend and hardware support.
- Memory, speed, and output-quality tradeoffs.

## Assignments

- [ ] Reuse Stage 1's CPU float32 and MPS float32 baselines with Stage 3's
  measurement procedure.
- [ ] On the same supported device, compare float32 with float16 and/or bfloat16
  without changing the model or decoding settings.
- [ ] Choose one supported quantization backend and verify model compatibility
  before installing additional packages.
- [ ] Compare memory usage.
- [ ] Compare performance on the same workload.
- [ ] Run the fixed quality-check prompts.
- [ ] Record numerical format, model revision, and backend.
- [ ] Explain which tensors are actually quantized.

Do not assume weight quantization also reduces KV-cache memory.

If quantization requires changing the execution backend, first establish an
unquantized baseline on that backend where possible. Otherwise, clearly identify
the comparison as confounded.

### Concrete Apple Silicon route

Start with the precision comparisons above. Then investigate either
`llama.cpp` with GGUF or `mlx-lm` with MLX quantized weights. Choose one,
check the current model and precision support, and keep a higher-precision
baseline on that same backend where available.

Do not assume CUDA-oriented package instructions, including a particular
bitsandbytes configuration, work on the Mac. If switching to a Linux GPU target,
make its compatible quantization path an explicit decision.

Verify the source checkpoint, tokenizer, and chat template of converted models.
A changed backend or converted checkpoint must be recorded, not attributed
solely to quantization. Keep these backend choices distinct from the vLLM/SGLang
serving-engine comparison in Stage 8.

## Acceptance criteria

- The chosen configuration actually runs on the target device.
- Claims separate memory savings from speed improvements.
- Output quality is inspected, not assumed.
- A slower quantized result is treated as a valid finding.

## Evidence

A comparison table covering configuration, memory, latency, throughput, and
observed output differences.

## Suggested commit

`Compare quantized and full-precision inference`

---

# Stage 8: Use an Inference Engine

## Central question

What does a specialized engine provide beyond my Transformers service?

## Scope

Choose one engine first:

- vLLM, or
- SGLang.

Do not integrate both simultaneously.

## Concepts

- Request scheduling.
- Continuous batching.
- KV-cache management.
- Engine-specific optimizations.
- Streaming contracts.
- Backend support.
- API compatibility.

## Assignments

- [ ] Check current hardware and operating-system support.
- [ ] Run the chosen engine with a supported model.
- [ ] Reuse the independent benchmark client and the supported
  `/v1/chat/completions` request/streaming subset.
- [ ] Compare a low-load case and a concurrent-load case.
- [ ] Document engine configuration.
- [ ] Identify which responsibilities moved from my code into the engine.
- [ ] Verify response fields, finish events, token usage, and parameter mapping
  rather than assuming "OpenAI-compatible" means identical behavior.

For Apple Silicon, evaluate the appropriate supported backend. Do not assume a
CUDA installation guide applies to the Mac.
Use the hardware decision from Stage 5. Pin engine version and launch options,
and match token IDs/chat templates, precision, EOS rules, and output controls.
Reuse both natural-stopping and fixed-length workloads with clear labels.

## Acceptance criteria

- The engine runs a real model, not a mock.
- Comparisons use the same hardware and model settings where possible.
- Differences in hardware, precision, or tokenization are disclosed.
- I can explain at least two mechanisms behind observed performance differences.

## Evidence

A baseline-versus-engine comparison and a short architectural explanation.

## Suggested commit

`Integrate an inference engine and compare serving behavior`

---

# Stage 9: Add Multiple Models and Routing

## Central question

How should a service choose a model without losing control of memory and
behavior?

## Concepts

- Model identifiers.
- Model registry.
- Explicit routing.
- Lazy versus eager loading.
- Resource limits.
- Model availability.
- Quality/latency tradeoffs.
- Failure behavior.

## Assignments

- [ ] Add explicit model selection to the request contract.
- [ ] Maintain an allowlist of supported models.
- [ ] Reject unknown model identifiers.
- [ ] Define how many models may be loaded simultaneously.
- [ ] Prevent simultaneous requests from triggering duplicate loads.
- [ ] Define behavior when the selected model is unavailable.
- [ ] Add one understandable routing policy only after explicit selection works.

Start with a simple policy, such as explicit user choice or a documented
latency/quality preference.

Do not begin with an LLM-powered router.

## Acceptance criteria

- Requests reach the intended model.
- Model selection does not permit arbitrary downloads or filesystem access.
- Memory remains bounded.
- Model-loading failures produce clear errors.
- Responses identify which model served the request.

## Evidence

Examples of successful routing, unknown-model rejection, and unavailable-model
handling.

## Suggested commit

`Add explicit multi-model routing with resource limits`

---

# Stage 10: Monitor and Deploy the Service

## Central question

How do I know the service is healthy, and how does it behave when something
fails?

## Concepts

- Structured logs.
- Metrics.
- Readiness and liveness.
- Graceful shutdown.
- Resource limits.
- Authentication.
- Reproducible deployment.
- Configuration and secrets.
- Operational documentation.

## Assignments

- [ ] Record request IDs and outcome status.
- [ ] Expose request counts and error counts.
- [ ] Expose latency, queue depth, and token throughput.
- [ ] Track memory with clearly defined measurements.
- [ ] Avoid logging private prompt contents by default.
- [ ] Handle process startup and shutdown deliberately.
- [ ] Provide a reproducible launch procedure.
- [ ] Add authentication before exposing the service beyond a trusted local
environment.
- [ ] Define input, output, queue, and concurrency limits.
- [ ] Document model-cache and credential handling.
- [ ] Test a deployment restart and a failed startup.

Containerization is optional until there is a concrete deployment target. If
used, verify hardware access and model-cache behavior rather than assuming
containers provide them automatically.

## Acceptance criteria

- A new environment can follow the documented setup.
- Readiness accurately reflects whether inference can be served.
- Failures are visible without exposing secrets or private prompts.
- Shutdown behavior is documented and exercised.
- The deployed configuration has bounded resource usage.

## Evidence

A deployment guide, a failure/recovery walkthrough, and a metrics view.
A dedicated dashboard is optional; useful observable signals are required.

## Suggested commit

`Add operational monitoring and reproducible deployment`

---

# 5. Cross-Cutting Quality Checks

## Small prompt collection

Maintain a small fixed set of prompts covering:

- A factual question.
- A formatting instruction.
- A short creative request.
- A simple reasoning task.
- A longer input.
- A request likely to reach the output limit.

This collection is a sanity check, not a comprehensive quality benchmark.

Record actual failures. Do not edit examples to make results look better.

## Functional checks

As features appear, verify:

- Output excludes the input prefix.
- Output length respects the configured limit.
- EOS stopping works.
- Invalid parameters are rejected.
- Batch outputs map to the correct inputs.
- Requests do not leak state into each other.
- Canceled requests stop consuming resources when cancellation takes effect.
- Queue overflow has defined behavior.
- Unknown models are rejected.
- Startup failure does not report readiness.

Automate checks when they protect meaningful behavior, especially API contracts
and scheduling logic.

Do not write tests that require a small language model to produce one exact prose
answer.

Keep API and scheduler checks fast by substituting a narrow deterministic
inference test double only at the execution boundary. Exercise real admission,
queueing, cancellation, and response-state transitions; do not merely assert
that a mock returns the value it was configured to return.

Retain real Qwen smoke runs for tokenizer behavior, actual generation, and the
end-to-end service. A fake backend cannot establish model correctness or
performance. Keep tests independent of incidental prose from the real model.

---

# 6. Benchmark Record Template

## Experiment name

### Question

What am I trying to learn?

### Hypothesis

What do I expect, and why?

### Environment

- Hardware:
- Execution device/backend:
- Python version:
- Library versions:
- Engine/backend version and launch settings:
- Chat template and tokenizer revision:
- Model ID:
- Model revision:
- Weight precision:
- KV-cache configuration:

### Workload

- Prompt set:
- Input token lengths:
- Output limits:
- Actual generated token counts:
- Decoding settings:
- Seeds:
- Concurrency:
- Batch size:
- Warm-up procedure:
- Number of measured requests:
- Natural-stopping or fixed-length workload:
- Timestamp boundaries and clock:
- Generated-token versus streamed-chunk counting:
- Errors, cancellations, and treatment of incomplete requests:
- Profiling enabled or disabled:

### Controlled change

What single variable changed?

### Results

| Configuration | E2E p50/p95 | Model TTFT | Client first text | Mean TPOT | Output tokens/s | Memory | Requests/errors |
|---|---:|---:|---:|---:|---:|---:|---:|

### Interpretation

What do the results support?

### Limitations

What do the results not prove?

### Next question

What should I investigate next?

---

# 7. Git Workflow

A commit should capture one meaningful, working learning checkpoint.

Examples:

- `Inspect next-token logits`
- `Add a manual greedy decoding loop`
- `Compare sampling settings`
- `Expose local inference through an API`
- `Add bounded request scheduling`

Before committing:

- [ ] Save the relevant files.
- [ ] Run the changed behavior.
- [ ] Update the README with important findings.
- [ ] Check that environments, weights, and secrets are excluded.
- [ ] Stage intentional files.
- [ ] Inspect the staged diff.
- [ ] Write a summary describing the change.
- [ ] Add a body when the motivation or experimental result needs explanation.

Do not describe downloaded weights or an activated shell environment as committed
artifacts unless the commit actually includes relevant setup code or
documentation.

---

# 8. Repository Organization

## Start simple

    inference-platform/
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    └── src/
        └── main.py

Keep `plan.md` as the roadmap. Add lab notes under `docs/` when needed; the
existing README remains the quick-start and a summary of demonstrated results.

## Grow only when responsibilities separate

Possible later structure:

    inference-platform/
    ├── README.md
    ├── requirements.txt
    ├── plan.md
    ├── src/
    │   ├── main.py
    │   ├── inference.py
    │   ├── api.py
    │   └── scheduler.py
    ├── benchmarks/
    ├── tests/
    └── docs/

Do not create every module now.

Extract a module when there is a clear responsibility, a real caller, and a
useful boundary.

---

# 9. Reading Map

## Model and tokenization

- [Qwen2.5-0.5B-Instruct model
card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Hugging Face tokenizer
lesson](https://huggingface.co/learn/llm-course/chapter2/4)
- [Hugging Face model-loading
lesson](https://huggingface.co/learn/llm-course/chapter2/3)
- [Chat templates](https://huggingface.co/docs/transformers/chat_templating)
- [Qwen model API](https://huggingface.co/docs/transformers/model_doc/qwen2)

## Generation

- [Text generation guide](https://huggingface.co/docs/transformers/llm_tutorial)
- [Generation
strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Generation parameter
reference](https://huggingface.co/docs/transformers/main_classes/text_generation)

## Serving and hardware

- [FastAPI tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI concurrency explanation](https://fastapi.tiangolo.com/async/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Transformers streamers and stopping criteria](https://huggingface.co/docs/transformers/internal/generation_utils)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)

## Optimization

- [KV-cache
explanation](https://huggingface.co/docs/transformers/cache_explanation)
- [Transformers quantization
overview](https://huggingface.co/docs/transformers/quantization/overview)
- [vLLM installation and hardware
support](https://docs.vllm.ai/en/stable/getting_started/installation/)
- [SGLang documentation](https://docs.sglang.ai/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [MLX language models](https://github.com/ml-explore/mlx-lm)
- [Qwen generation defaults](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/generation_config.json)

Read the section needed for the current experiment. Do not try to finish every
reference before writing code.

---

# 10. Immediate Next Assignment

## Assignment: Inspect One Next-Token Prediction

Do this before starting the API.

### Tasks

1. Reuse the working model and tokenized input.
2. Run one forward pass without `generate()`.
3. Print the logits shape.
4. Select the next-token logits.
5. Convert them to probabilities.
6. Display the five highest-probability candidates.
7. Check that probabilities across the full vocabulary sum approximately to one;
   the displayed top five generally will not.

### Bring to the review

- The code I added.
- The actual logits shape.
- The five candidates and probabilities.
- My explanation of the tensor dimensions.
- My explanation of why I selected the final input position.
- My prediction of which token greedy decoding would choose.

### Completion condition

I can explain the result without relying on “the library does it.”

Use a short fixed prompt and evaluation/inference mode. Do not interpret token
probability as the probability that an eventual answer is true.

Only then move on to choosing a token and building the short manual generation
loop.
