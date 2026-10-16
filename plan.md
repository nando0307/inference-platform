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
- Explain proposed file changes and wait for explicit confirmation before editing.
  Pasted results or a request to continue learning are not edit permission.

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

Use `plan.md` as the stable roadmap. The active checkpoint, commands, results,
and open questions live in [logs/practice.md](logs/practice.md). Read that handoff
before resuming a session; roadmap checkboxes are not a live completion ledger.
Do not create empty documentation files in advance.

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

## Scope and baseline

Build a localhost HTTP service that loads `Qwen/Qwen2.5-0.5B-Instruct` once,
supports ordinary and streaming chat completions, validates requests, rejects
overlapping inference, and recovers correctly from errors and disconnected
clients.

This is a learner-driven roadmap, not a record of completed implementation.
Work through one checkpoint at a time and verify its completion criteria before
continuing. Discuss the first coding task and obtain permission before source
edits or package installation. Updating this roadmap does not authorize
implementing the service.

- Keep CPU float32 for the initial server; MPS remains a separate experiment.
- Use Hugging Face `generate()` with `use_cache=True`.
- Preserve all four educational scripts. Do not import an executable experiment
  module into the application.
- Use one process, one loaded model, and one admitted producer at a time.
- Bind to localhost only. No containers, authentication system, persistent
  conversation store, frontend, or generic backend interface are needed.
- Support at most 2048 total formatted tokens and 512 new tokens per request.
- Reproducible benchmarking belongs to Stage 3. Bounded request queues,
  cooperative cancellation, and slow-consumer policies belong to Stage 4.

## Checkpoint 1: Understand the service boundary

**Question:** What changes when a script serves many requests without exiting?

- [ ] Trace the current script: load model, format prompt, tokenize, generate,
  print, exit.
- [ ] Trace the service: start process, load once, accept requests, validate,
  acquire the inference slot, prepare input, generate, return a result, repeat.
- [ ] Identify process-lifetime state: model weights, tokenizer, model
  configuration, one execution worker, and admission guard.
- [ ] Identify request-local state: messages, input tensors, generation
  settings, generated IDs, KV cache owned by that generation call, response ID,
  timing, and streaming queue.
- [ ] Keep the initial CPU float32 baseline and use library generation rather
  than implementing another sampling or attention loop.

**Explain before continuing:** Weights are shared learned parameters. Generated
sequences and KV caches encode a particular request's context; reusing them
across unrelated requests would mix conversations.

**Done when:** You can draw the request path and explain each object's lifetime.

## Checkpoint 2: Extract reusable inference

**Question:** What should inference return instead of printing?

- [x] Create `src/inference.py`, independent of HTTP and terminal input.
- [x] Define `load_runtime() -> Runtime`; load weights only inside this function.
- [x] Define a small `Runtime` dataclass with `model`, `tokenizer`,
  `eos_token_ids: set[int]`, and `context_limit: int`.
- [x] Define a small `GenerationResult` dataclass with `text: str`,
  `prompt_tokens: int`, `generated_ids: list[int]`, and
  `finish_reason: Literal["stop", "length"]`.
- [x] Define `ContextLimitError` for oversized requests.
- [x] Implement this reusable boundary:

```python
generate(
    runtime,
    messages,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    streamer=None,
    on_prepared=None,
) -> GenerationResult
```

Reuse the service-relevant loading, template, device, EOS-normalization, and
output-slicing operations from `src/cached_generation.py`, not its experiment,
timers, console output, or parity assertions.

- [x] Keep CPU execution, `dtype=torch.float32`, and `model.eval()`.
- [x] Enter `torch.inference_mode()` inside the function executed by the worker.
  The context is thread-local; wrapping only the HTTP caller is insufficient.
- [x] Apply the chat template with `tokenize=False` and
  `add_generation_prompt=True`.
- [x] Tokenize with `add_special_tokens=False`, an attention mask, and no
  truncation. Move tensors to the model's device.
- [x] Pass messages through without adding an application-defined system
  message. Any template-added system text remains tokenizer behavior.
- [x] Validate the combined context budget before generation.
- [x] Invoke the optional no-argument `on_prepared()` callback only after input
  preparation and context validation succeed, before generation.
- [x] Slice `output_ids[0, prompt_length:]` so completion text excludes input.
- [x] Decode with `skip_special_tokens=True` and
  `clean_up_tokenization_spaces=False`; streaming must use the same settings.
- [x] Count generated IDs, including EOS, rather than visible text.
- [x] Report `stop` if the final generated ID is EOS, including at the cap;
  otherwise report `length` when the output budget is exhausted.
- [x] Let unexpected failures propagate rather than returning empty success.

**Exercise:** Load one runtime and call `generate()` twice. Confirm there is no
interactive prompt and each result contains only its own completion.

**Done when:** Importing the module does not load weights, and both calls reuse
one runtime.

## Checkpoint 3: Define the request contract

**Question:** What does the service promise, and what must it reject?

- [ ] Create Pydantic request models in `src/api.py`.
- [ ] Define `ChatMessage` with roles `system`, `user`, or `assistant` and
  nonempty string content. Reject whitespace-only content but retain valid text
  exactly as supplied.
- [ ] Define `ChatCompletionRequest` using the following rules.
- [ ] Reject extra fields at both request and message levels.
- [ ] Require at least one message without imposing alternating-role rules.

| Field | Rule | Default |
|---|---|---|
| `model` | Exactly `Qwen/Qwen2.5-0.5B-Instruct` | Required |
| `messages` | Nonempty array of supported messages | Required |
| `max_tokens` | Integer from 1 through 512 | `100` |
| `temperature` | Finite number from 0 through 2 | `0` |
| `top_p` | Finite number greater than 0 and at most 1 | `1` |
| `stream` | Boolean | `false` |

- [ ] Reject numeric strings, booleans used as numbers, nonfinite values, and
  nonboolean `stream` values. Ordinary JSON integers remain valid temperature
  and top-p numbers.
- [ ] Reject tools, image content, `n`, client `top_k`, `stop`, `seed`,
  `stream_options`, and all other unsupported fields.
- [ ] Until Checkpoint 9, reject `stream=true` with HTTP 422 and code
  `unsupported_parameter`; remove this temporary restriction when streaming
  is implemented.

This is a documented chat-completions subset, not full OpenAI API compatibility.

**Explain before continuing:** Parsing a field is not implementing its behavior.
Accepting and ignoring a setting falsely promises support.

**Done when:** Missing, malformed, unknown, and unsupported fields have
predictable rejection behavior.

## Checkpoint 4: Map settings to generation explicitly

**Question:** How does an API parameter change generation?

- [ ] Construct a fresh `GenerationConfig` per request. Never mutate shared
  `model.generation_config`.
- [ ] Map external `max_tokens` to internal `max_new_tokens`.
- [ ] Set `use_cache=True`, `num_beams=1`, `num_return_sequences=1`, and
  `repetition_penalty=1.0`.
- [ ] Preserve the configured EOS IDs from `model.generation_config`, not only
  the single EOS ID from `model.config`.
- [ ] Use the tokenizer's pad ID, falling back to the first configured EOS ID.
  If neither exists, fail startup clearly rather than guessing.

| API input | Library-facing settings |
|---|---|
| `temperature == 0` | `do_sample=False`, `temperature=1.0`, `top_p=1.0`, `top_k=50` |
| `temperature > 0` | `do_sample=True`, requested temperature/top-p, `top_k=0` |

- [ ] Reject temperature zero with `top_p != 1.0` as HTTP 422
  `invalid_parameter_combination`; top-p does not change greedy decoding.

The inspected Qwen snapshot `7ae557604adf67be50417f59c2c2f167def9a775` saves
sampling enabled, temperature `0.7`, top-p `0.8`, top-k `20`, and repetition
penalty `1.1`. Override all five as specified. Preserve both EOS IDs,
`[151645, 151643]`. Do not suppress configuration warnings or edit pretrained
files. If the snapshot changes, inspect its settings and explicitly neutralize
new sampling/logit modifiers that would change the documented contract.

**Exercises:**
- Repeat identical greedy inputs on the same runtime and compare generated IDs.
- Generate with temperature `0.7` and top-p `0.9`.
- Reject temperature `0` with top-p `0.9`.
- Do not require sampled outputs to differ; equal samples can occur.

**Done when:** Every accepted parameter has an explained implementation without
invalid-generation-flag warnings.

## Checkpoint 5: Enforce the combined context budget

**Question:** Why can a short-looking message exceed a token budget?

- [ ] Set `context_limit = min(2048, model.config.max_position_embeddings)`.
- [ ] Fail startup if the declared model limit is absent or invalid. Do not
  guess a limit or apply positional-extension settings.
- [ ] After formatting and tokenization, require:

```text
prompt_tokens + requested_max_tokens <= context_limit
```

Count the whole formatted conversation, role markers, template-added system
text, and assistant generation prefix. The 2048-token limit is local serving
policy, not a claim about Qwen's architectural maximum.

- [ ] Return HTTP 400 `context_length_exceeded` when the combined budget exceeds
  the limit.
- [ ] Include prompt count, requested output count, and supported limit in the
  error, without echoing input.
- [ ] Do not truncate messages, discard history, or reduce the output cap.
- [ ] Accept exact fits and reject one token over.
- [ ] Validate before `model.generate()` and before SSE headers are committed.

**Exercise:** Use the actual tokenizer to find a near-boundary prompt. Test an
output cap that exactly fits, then increase it by one. Repeat for streaming
after Checkpoint 9.

**Done when:** Actual formatted-token counts enforce both boundaries.

## Checkpoint 6: Give the application a lifecycle

**Question:** When is the process alive, and when can it perform inference?

- [ ] Add `fastapi` and `uvicorn` to `requirements.txt` during implementation.
  Keep the existing `.venv` and its PyTorch/Transformers versions.
- [ ] Create `app = FastAPI(lifespan=lifespan)` in `src/api.py` using
  `contextlib.asynccontextmanager`, not import-time loading or deprecated event
  decorators.
- [ ] Store runtime, worker, admission guard, and shutdown state on `app.state`.

Startup:
1. Initialize state as unavailable.
2. Load tokenizer and model exactly once through `load_runtime()`.
3. Create `ThreadPoolExecutor(max_workers=1)` and a `threading.Lock`.
4. Mark initialized only after successful loading.
5. Propagate startup failure; do not serve a healthy-looking app without a model.

| Endpoint/state | Status | Body |
|---|---|---|
| `GET /health`, serving HTTP | 200 | `{"status":"alive"}` |
| `GET /ready`, initialized and idle | 200 | `{"ready":true,"status":"ready"}` |
| `GET /ready`, busy | 503 | `{"ready":false,"status":"busy"}` |
| `GET /ready`, shutting down, if reachable | 503 | `{"ready":false,"status":"shutting_down"}` |

Uvicorn may refuse connections while lifespan startup loads weights. Connection
refusal is not an HTTP 503 response; do not claim a pre-start response unless
one was actually received.

Shutdown:
1. Mark unavailable and prevent new admissions.
2. Wait for the admitted producer to finish.
3. Shut down the executor, then drop runtime references.
4. Never unload the model while its worker is executing.

**Exercise:** Start the service, wait for readiness, send two requests, verify
one startup load, and stop it normally.

**Done when:** Model lifetime matches application lifetime.

## Checkpoint 7: Serialize inference without blocking HTTP

**Question:** Why does `async def` not make model computation asynchronous?

Implement this before accepting the first generation request.

- [ ] Perform cheap schema validation before admission.
- [ ] Return HTTP 503 `not_ready` if uninitialized or shutting down.
- [ ] Acquire the guard with `acquire(blocking=False)` before submitting work.
- [ ] Return HTTP 429 `server_busy` immediately if occupied.
- [ ] Submit only admitted work to the dedicated executor. Its internal queue
  must not become an application request queue.
- [ ] Run tokenization, context validation, and generation in the producer,
  keeping shared tokenizer use serialized and off the event loop.
- [ ] Retain the producer future until actual completion, shield it from HTTP
  cancellation, and observe exceptions even if the client disconnects.
- [ ] Release admission in the producer's `finally`, after its work exits.
- [ ] If submission fails before the producer starts, release the guard in the
  submitting path and return HTTP 500 `generation_failed`.
- [ ] For streaming, maintain a per-request preparation signal resolved by
  `on_prepared()` or by preparation failure. Never leave it unresolved.
- [ ] Await preparation before returning `StreamingResponse`.

Do not call blocking generation, queue reads, or thread joins on the event loop.
Do not use an `asyncio.Lock` that makes excess requests wait.

**Exercises:** During a long generation, verify health remains 200, readiness is
503 busy, and a second valid generation receives 429. Verify readiness returns
after actual producer completion.

**Done when:** At most one producer uses the model/tokenizer while HTTP remains
responsive.

## Checkpoint 8: Return ordinary completions and useful errors

**Question:** How does a model result become an HTTP response?

- [ ] Implement async `POST /v1/chat/completions` through the admission/worker
  path, not by calling blocking generation directly.
- [ ] For `stream=false`, return HTTP 200 with:
  - `id`: `chatcmpl-` followed by UUID hex.
  - `object`: `chat.completion`.
  - `created`: integer Unix timestamp.
  - `model`: the fixed supported model ID.
  - One choice with `index: 0`, assistant message content, and finish reason
    `stop` or `length`.
  - Integer `usage.prompt_tokens`, `usage.completion_tokens`, and
    `usage.total_tokens`.
- [ ] Compute completion tokens from `len(result.generated_ids)` and total
  tokens from prompt plus completion. Do not report the requested cap as actual
  usage when EOS stopped early.
- [ ] Use this consistent error envelope:

```json
{
  "error": {
    "code": "context_length_exceeded",
    "message": "A useful description without private input.",
    "request_id": "..."
  }
}
```

| HTTP status | Code and cause |
|---|---|
| 422 | `invalid_request`: malformed JSON, missing/unknown fields, invalid types/ranges, unsupported model or roles |
| 422 | `invalid_parameter_combination`: incompatible supported settings |
| 422 | `unsupported_parameter`: streaming before Checkpoint 9 |
| 400 | `context_length_exceeded`: combined token budget exceeded |
| 429 | `server_busy`: an admitted producer owns inference |
| 503 | `not_ready`: unavailable or shutting down |
| 500 | `generation_failed`: unexpected preparation/inference failure |

- [ ] Customize FastAPI validation errors to omit raw inputs and bodies.
  Locations and constraints may be included, but not private content.
- [ ] Never expose raw unexpected exception text or tracebacks in responses.
- [ ] Preserve normal 404/405 routing behavior.

**Exercises:** Successful curl request, unsupported model, empty messages,
negative token cap, unknown field, early EOS, and a deliberately small cap.

**Done when:** Ordinary HTTP generation works with truthful usage and finish
reasons.

## Checkpoint 9: Stream text before generation finishes

**Question:** Are you streaming tokens, text, or HTTP events?

- [ ] Remove the temporary rejection of `stream=true`.
- [ ] Reread the installed `TextIteratorStreamer` implementation before building
  the consumer.
- [ ] Create one streamer per request with `skip_prompt=True`,
  `skip_special_tokens=True`, `clean_up_tokenization_spaces=False`, and
  `timeout=0.1`.
- [ ] Pass it into the same admitted producer and reusable `generate()`.
  Do not introduce a second model-execution thread or generation algorithm.
- [ ] Read its blocking queue off the event loop.
- [ ] Translate iterator outcomes into plain text/timeout/end values; do not
  propagate `StopIteration` through an asyncio future.
- [ ] On timeout, recheck producer completion/error and client connectivity.
  Timeout alone is not failure.
- [ ] End promptly on producer failure, even without a streamer sentinel.
- [ ] If the sentinel arrives first, await the producer result before declaring
  successful termination.
- [ ] Skip empty text chunks and count tokens from producer-generated IDs.
  Do not re-tokenize streamed text or independently decode each token.

Use `Content-Type: text/event-stream`, `Cache-Control: no-cache`, and frames
formatted as `data: <JSON>\n\n`.

Successful stream contract:
1. Initial chunk: `object: "chat.completion.chunk"` with
   `choices: [{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]`.
2. Content chunks: newly available text in `choices[0].delta.content`, with null
   finish reason.
3. Terminal chunk: empty delta, actual finish reason, and top-level `usage`
   matching ordinary generation counts.
4. Exactly one final frame: `data: [DONE]\n\n`.

Keep ID, model, and creation timestamp consistent throughout. Terminal usage is
an explicitly documented local extension; `stream_options` remains unsupported.

**Exercises:**
- Use `curl -N` and observe progressive output.
- Prove nonempty content reaches the client before the producer finishes; the
  initial role event alone is not evidence of useful streaming.
- Compare concatenated streamed text with ordinary output for identical greedy
  inputs on the same runtime and with matching decode settings.
- Include multilingual text and punctuation to check incremental decoding.
- Inspect chunk boundaries: chunks are not token counts, and some tokens do not
  immediately produce printable text.

**Done when:** Incremental content arrives before completion and ends with
truthful metadata and one `[DONE]`.

## Checkpoint 10: Handle failure and disconnect ownership

**Question:** Who owns inference after a client disappears?

- [ ] Before headers, return ordinary HTTP errors using Checkpoint 8.
- [ ] After headers, do not attempt to rewrite HTTP status. On producer failure,
  emit the same error envelope as SSE with code `generation_failed`, if still
  connected, followed by `[DONE]`.
- [ ] Do not emit a successful terminal finish reason or fabricated usage for
  incomplete generation.
- [ ] On disconnect, stop delivering output but retain admission until the
  producer actually exits.
- [ ] Let finite generation finish under existing caps, discard undelivered
  output, reclaim request state, and observe any producer exception.
- [ ] Do not spawn another model call because the HTTP coroutine was canceled.
- [ ] Verify a later request succeeds after a failed producer exits.

The library queue is not intrinsically bounded, but this stage admits only one
producer with at most 512 generated tokens. Explicit slow-consumer backpressure
and cooperative stopping belong to Stage 4.

A permanently stuck native call cannot safely be killed by canceling a Python
future. Leave inference unavailable and require process restart; do not fake
recovery or launch a replacement producer over the same model.

**Exercises:** Disconnect after the first content event. Confirm busy rejection
while inference remains active and success afterward. Inject a controlled
generation failure; confirm stream termination without a missing-sentinel hang,
then successful recovery. Use a deterministic held producer in regression tests
to prove ownership rather than relying on a race with natural completion.

**Done when:** Neither failures nor disconnects leak ownership or release it
before model work ends.

## Checkpoint 11: Add request IDs and private lifecycle logging

**Question:** How do you diagnose failures without saving conversations?

- [ ] Use standard Python logging with JSON records.
- [ ] Generate a server-side UUID per chat request; do not trust an arbitrary
  incoming identifier.
- [ ] Return `X-Request-ID` and include the ID in errors and lifecycle logs.
- [ ] Record acceptance/rejection and final outcome.
- [ ] Record model, stream mode, known HTTP status, monotonic duration in
  milliseconds, and outcome: `completed`, `rejected`, `failed`, or `disconnected`.
- [ ] Record actual prompt/output counts when available; never fabricate them.
- [ ] Start timing before validation/admission and finish successful delivery
  timing after the ordinary body or final SSE event is sent.
- [ ] Distinguish HTTP disconnect time from eventual producer completion.
  Measure any generation-duration field separately around the producer.
- [ ] For unexpected failures, record exception class and sanitized error code,
  not arbitrary exception text.

Never log messages, rendered prompts, generated text, token IDs, raw bodies,
authorization headers, or raw validation-input values. These are lifecycle
measurements, not Stage 3 TTFT/TPOT benchmarks.

**Exercise:** Send an unmistakable private marker in valid and invalid requests.
Inspect logs for request correlation and status while confirming the marker and
generated content are absent.

**Done when:** Requests are diagnosable without exposing conversation content.

## Checkpoint 12: Prove the complete service

- [ ] Add permanent behavioral regressions in `tests/test_api.py` using the
  standard-library `unittest` runner; add `httpx` only if needed as a test client.
- [ ] Cover combined context exact-fit and one-token-over boundaries, including
  template overhead.
- [ ] Cover invalid numeric types, nonfinite values, unknown fields, and
  incompatible greedy/top-p settings.
- [ ] Hold a producer and prove 429 rejection while health remains responsive.
- [ ] Prove consumer cancellation/disconnect does not admit work before actual
  producer exit.
- [ ] Prove failure before preparation resolves the waiter as an ordinary error.
- [ ] Prove failure after stream start terminates without a successful finish
  event and permits a later request.
- [ ] Prove EOS at the output cap reports `stop`.
- [ ] Prove private inputs are absent from validation responses and logs.
- [ ] Run the actual application with the real Qwen model over HTTP. Held or
  failing fixtures prove transitions, not real-model integration.
- [ ] Update existing run/contract documentation and `logs/practice.md` with
  observed evidence after implementation. Do not manually edit generated
  changelogs.

### Environment and launch

The following are future implementation checks, not claims that a server exists
or these checks have passed. Run from the repository using its existing `.venv`.
After dependency edits are authorized and applied:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn src.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Do not use reload or multiple worker processes for evidence collection. Load
the real model from cache or download it; never substitute fake answers if
loading fails.

### Ordinary success and readiness

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Explain prefill in two sentences."}],"max_tokens":100,"temperature":0,"top_p":1,"stream":false}'
```

Expect health 200, idle readiness 200, and completion 200 with assistant text,
request ID, correct counts, and `stop` or `length`. Repeat without another model
load. Do not assert exact natural-language wording.

### Invalid settings and oversized input

```bash
curl -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello"}],"temperature":0,"top_p":0.9}'

.venv/bin/python -c 'import json; print(json.dumps({"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"hello "*5000}],"max_tokens":100}))' | \
  curl -i http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary @-
```

Expect 422 `invalid_parameter_combination` for the first and 400
`context_length_exceeded` for the second, without echoed input. Repeat oversized
input with `stream=true`; expect ordinary HTTP 400, not a late SSE validation
error. Separately prove exact-fit and one-token-over boundaries with the actual
tokenizer. Also verify a valid sampling request with temperature `0.7`, top-p
`0.9`.

### Streaming, overlap, and disconnect

In terminal A:

```bash
curl -N -i http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Write a detailed numbered explanation of how an inference service processes a request."}],"max_tokens":512,"temperature":0,"top_p":1,"stream":true}'
```

During generation, use terminal B for health, readiness, and the ordinary
completion command. Expect health 200, ready 503 busy, and completion 429.
Observe nonempty streamed content before producer completion, then truthful
terminal usage/finish reason and one `[DONE]`. Verify readiness and a new
completion succeed afterward.

Repeat and interrupt terminal A after content starts. Verify ownership remains
until producer exit, then availability returns. A short remaining generation
may finish before the next request arrives; use the held-producer regression
to prove the invariant rather than treating that race as failure.

Compare ordinary and concatenated streamed greedy text on the same runtime,
including multilingual text and punctuation.

### Controlled failure and regression run

Use a temporary harness with the actual application and loaded Qwen model.
Wrap one `model.forward` call to raise a controlled `RuntimeError` after at
least one successful decode step. Observe stream termination and slot recovery,
restore the method in `finally`, and send a successful real request afterward.
Do not add a public failure-injection endpoint or production debug flag.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

### Evidence to retain

Record actual settings, status codes, token counts, finish reasons, and ownership
transitions for ordinary success, invalid settings, oversized input, streaming,
overlap, disconnect, and producer failure. Distinguish visible-first-text timing
from first generated-token timing. A streamer chunk is not a token, and these
single runs are not benchmarks.

If weights/network access are unavailable, finish reachable validation/protocol
checks and report the missing artifact. Do not claim end-to-end completion or
silently substitute a different model.

### Learning checks

Answer without referring to implementation:

1. Why load the model once rather than per request?
2. Why does importing the application not load weights?
3. How do `model.eval()` and `torch.inference_mode()` differ?
4. Why format messages before counting tokens?
5. How does external `max_tokens` map to internal `max_new_tokens`?
6. What does temperature zero mean in this API?
7. Why is top-p inactive for greedy decoding?
8. Why can visible text and completion-token count differ?
9. Why does `async def` not prevent blocking computation?
10. Why does a single-worker executor still need admission control?
11. How do liveness, readiness, and busy status differ?
12. Why is a text chunk not a token?
13. What changes after streaming headers are committed?
14. Why does disconnect not necessarily stop computation?
15. Why must ownership last until producer exit?

## Acceptance criteria

Stage 2 is complete only when all of the following are demonstrated:

- Multiple requests reuse one loaded model.
- Ordinary and streaming HTTP requests follow the documented contract.
- All stated validation rules hold, including combined formatted-token budgets.
- Usage and finish reasons reflect actual generated IDs.
- Readiness is not claimed before successful loading or while inference is busy.
- Overlapping inference is rejected while health remains responsive.
- Nonempty streamed content reaches the client before generation finishes.
- Producer failure terminates the stream and allows recovery after worker exit.
- Disconnect does not release admission while model work is still running.
- Request IDs and lifecycle logs support diagnosis without conversation content.
- Actual Qwen-over-HTTP smoke evidence complements behavioral regressions.

A working non-streaming endpoint is an intermediate checkpoint, not completion
of this stage.

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

# 10. Current Checkpoint and Next Assignment

Read [the session handoff in logs/practice.md](logs/practice.md) first.
The earlier one-token prediction assignment has been completed, followed by
manual greedy generation, basic sampling, Q/K/V inspection, cache reuse, and
a first single-run cached-versus-uncached timing comparison.

The learner stopped after the 34-token-prompt, 100-output-token comparison
to prepare a Git checkpoint. The measured results and exact timer boundaries
are recorded in the handoff. Do not restart the one-token exercise or assume
that every Stage 1 acceptance criterion has been completed.

The next proposed assignment is a controlled longer-prompt comparison, holding
model, CPU float32, greedy decoding, EOS rules, and output limit fixed.
It has not been implemented or measured. Discuss the prediction first and
obtain explicit confirmation before modifying files.

