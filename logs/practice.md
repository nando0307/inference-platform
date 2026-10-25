```text
# inference platform
Hi i am Do, i am learning about the inference !
my dream is to becoming a llm engineer xD
```


# Session handoff: 2026-10-14

## Start here, next mentor

The learner has completed manual cached and uncached generation, short/long
prompt timing comparisons, KV tensor storage inspection, token-ID comparisons
against Hugging Face `generate()` for token-limit and EOS stopping, and an
initial synchronized CPU/MPS float32 comparison. Do not restart the basics.

Current API state: Stage 2 technical acceptance, Phase 3 benchmarks, and Phase 4
bounded scheduling/cancellation are implemented. There are 106 passing tests.
Real-Qwen controls verify FIFO admission, queued removal, running disconnects,
bounded streaming pressure, request-setting isolation, and both shutdown paths.
See [the Phase 4 entry](#phase-4-bounded-concurrency-and-cancellation) for current
policy, measured concurrency evidence, and limitations.
Next: explain queue wait versus model time, cancellation ownership, and why
more connected clients do not imply parallel model execution. Independent
understanding checks remain pending. Earlier entries describe their own time.
Manual top-k/top-p lessons were explicitly skipped, not completed. Use the
stable roadmap [plan.md](../plan.md), this evidence log, and actual code together.

## Mentoring and edit permissions

- Teach inference engineering through Build -> Predict -> Measure -> Explain.
- Give small, bounded assignments; prefer hints and predictions before complete
  implementations unless the learner asks for code.
- Explain tensor shapes, data flow, and why a mechanism is needed.
- Ask for explicit confirmation before modifying any files. Pasted output,
  a conceptual question, or a request to continue learning is not edit approval.
  Describe the files and changes first and wait for confirmation.
- The learner explicitly declined repeated runs and median calculations for
  this experiment. Respect that choice and label results as single-run
  observations, not statistically stable benchmarks.
- Move toward the minimal local API next; do not jump to vLLM, CUDA,
  multi-worker infrastructure, or a larger architecture yet.
- This project runs Qwen locally, not through an OpenAI inference API.

## Current scripts

- `src/main.py`: tokenizer and embedding inspection, model dimensions,
  final-position logits/probabilities, top-five candidate display, and two
  manual greedy tokens without a KV cache.
- `src/architecture.py`: embedding lookup, first-layer input normalization,
  and Q/K/V projections. No full attention computation or cache construction.
- `src/uncached_generation.py`: renamed from `src/autoregressive.py`; menu
  choice `1` is greedy, `2` is temperature-0.7 multinomial sampling. Recomputes
  the growing prefix with `use_cache=False`. Includes prompt warm-up and
  separate prompt/subsequent-generation timers and a greedy `generate()`
  token-ID comparison. Choose menu option `1` for that comparison; the appended
  reference is greedy even if the manual path's sampling option is chosen.
- `src/cached_generation.py`: greedy generation; full-prompt prefill followed
  by single-token forwards passing `past_key_values`, with a full growing
  attention mask. Includes matching warm-up and timing boundaries, first-layer
  K/V shape/dtype inspection, and total KV tensor bytes before and after generation.
  It now selects `device = "cpu"` or `"mps"` while retaining float32, synchronizes
  MPS at timer boundaries, and compares output IDs against cached `generate()`.
  The current saved device is `"cpu"`.
- Both generation scripts use a 100-new-token limit and the model's configured
  EOS IDs, decode the generated IDs together, and print outside the timed loop.
- Both scripts currently use `Reply with exactly one word: Hello`, which yields
  28 chat-formatted prompt tokens. The 34-token and 516-token prompts below are
  historical experiment inputs, not the current input.
- `src/inference.py`: one loaded runtime, fresh generation settings per request,
  context checks, and cooperative cancellation around real generation.
- `src/api.py`: validated HTTP/SSE delivery, one active job, a two-slot FIFO,
  capacity-based readiness, lifecycle logging, and shutdown ownership.
- `src/streaming.py`: bounded Transformers text streamer with cancellation-aware
  backpressure; no alternate decoding implementation.
- `benchmarks/`: fixed workloads, bounded-wave HTTP clients, opt-in model and
  scheduling measurements, reporting, and separate model-side profiling.

## Concepts covered and demonstrated

- IDs index embeddings; logits are vocabulary scores, not probabilities.
- Prefill processes the full prompt and supplies logits for selecting token 1.
- Later cached forward passes process one new token using previous K/V state.
- Selecting a token does not grow the cache; processing it through the model does.
- Causal attention makes earlier K/V reusable when appending tokens: earlier
  positions cannot attend to future tokens. Unchanged token IDs alone are not
  the full explanation.
- The learner described reuse as recursion with memory. We refined this to
  memoization-like reuse within an iterative generation loop, not recursion.
- Cached decoding still attends to earlier context; one input position does not
  imply that increasing context length is free.
- Observed Qwen dimensions: hidden size 896, 24 layers, 14 query heads, 2 KV
  heads, 64 dimensions per head. GQA groups seven query heads per KV head.
- Prompt projection shapes: Q `[1, 34, 896]`, K/V `[1, 34, 128]`, before
  head reshaping and rotary positional embeddings.

## Experiment 1: 34-token cached versus uncached baseline

Configuration:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, Transformers/PyTorch, CPU float32.
- System message: `You are Qwen, a helpful assistant`.
- User prompt: `Hello world! Which paper mark the turning point for AI research?`
- Chat template uses `add_generation_prompt=True`; model input has 34 tokens.
- Greedy decoding in both scripts; 100 generated tokens, stopped at the limit.
- Evaluation mode and `torch.inference_mode()` are used.

The manual cached and uncached paths produced identical 100-token ID sequences
in an agent comparison, also consistent with the learner's pasted traces.
The response starts `The turning` and ends `in the field`.
The historical claims in the generated answer are hallucinated; token agreement
demonstrates generation equivalence for this case, not factual answer quality.

Paired baseline observations reported by the learner:

| Measurement | Cached | Uncached |
|---|---:|---:|
| Prompt forward pass | 96.00 ms | 91.11 ms |
| Subsequent generation | 7427.87 ms | 13658.61 ms |
| Tokens after the first | 99 | 99 |
| Average subsequent time/token | 75.03 ms | 137.97 ms |
| Subsequent throughput | 13.33 tokens/s | 7.25 tokens/s |

Observed subsequent-generation speedup: `13658.61 / 7427.87 = 1.84x`.
Observed time reduction: `1 - 7427.87 / 13658.61 = 45.62%`.
These results are not whole-request throughput or end-to-end latency.

Timing boundaries:

- One untimed prompt forward pass per script; its outputs are discarded.
- Prompt timer surrounds only the next full-prompt model call. This is not full
  TTFT because first-token selection and delivery are excluded.
- Subsequent timer starts after token 1 is selected and ends after the final
  token is selected and its stop condition checked.
- For 100 outputs this includes 99 forward passes, token selection, mask
  updates, loop overhead, and input-ID concatenation in the uncached path.
- Model loading, tokenization, menu input, printing, and final text decoding
  are outside both intervals.
- Prompt-only warm-up does not separately warm up the single-token decode path.
- Final cached length is 133: 34 prompt tokens plus 99 processed output tokens.
  The final selected token is not processed after stopping.

Earlier first-forward timing was 683.86 ms; after a prompt warm-up the learner
observed 95.05 ms. That was not a cache-on/cache-off comparison and must not be
reported as a KV-cache speedup.

## Experiment 2: prompt length

### Question and configuration

How does a longer prompt affect prompt-processing time and subsequent generation?
Keep Qwen 0.5B, CPU float32, batch size 1, greedy decoding, chat formatting,
EOS rules, and the 100-token output limit fixed. Change the prompt identically
in both scripts. Prompt content changes too, so this is not a perfectly isolated
measurement of length alone.

The learner ran `python src/uncached_generation.py` with menu choice `1`, then
`python src/cached_generation.py`. Both reported 516 input tokens and 99 tokens
after the first, stopping at the 100-token limit. The pasted responses matched;
at this point no programmatic long-prompt token-ID equality check was recorded.
Experiment 4 later verified both manual paths against Hugging Face on this prompt.

Actual prompt used in both scripts:

```python
prompt = (
    "Read the following background information.\n\n"
    + (
        "An inference service receives a request, tokenizes the prompt, "
        "runs a language model, and returns generated tokens. "
        "The service must balance response latency, throughput, and memory use."
        "A longer context gives the model more information to attend to, "
        "but also increases the amount of attention state involved in generation.\n"
    )
    * 8
    + "\nExplain how an inference service processes a request, "
    "from receiving the prompt to returning the generated response. "
    "Give a detailed explanation."
)
```

The adjacent strings intentionally preserve the text actually measured:
`memory use.A longer` has no intervening space. Do not silently normalize it
when reproducing these results. Chat-template tokens are included in the 516.

### Original timing comparison

All subsequent-generation intervals cover 99 tokens after the first.

| Measurement | 34-token baseline | 516-token prompt |
|---|---:|---:|
| Cached prompt forward | 96.00 ms | 389.11 ms |
| Uncached prompt forward | 91.11 ms | 391.61 ms |
| Cached subsequent generation | 7427.87 ms | 6786.19 ms |
| Uncached subsequent generation | 13658.61 ms | 49683.59 ms |
| Observed subsequent-generation speedup | 1.84x | 7.32x |

Long-prompt average subsequent time/token: cached 68.55 ms, uncached 501.85 ms.
Long-prompt subsequent throughput: cached 14.59 tokens/s, uncached 1.99 tokens/s.
Final cached length: `516 + 99 = 615`.

### Interpretation and limitations

- Prompt length grew about 15.18x, while prompt-forward time grew about 4x.
  Token count is not a constant-cost runtime multiplier: matrix shapes,
  utilization, and fixed overhead matter. These runs do not isolate their
  individual contributions.
- Uncached subsequent generation took 3.64x as long as the short-prompt run.
  It recomputed sequences of 517 through 615 positions instead of 35 through 133.
- Cached subsequent calls still processed one new position each. Avoiding
  repeated prefix computation became more valuable: observed speedup grew
  from 1.84x to 7.32x.
- Cached subsequent time was 8.64% lower in this particular long-prompt run.
  This does not prove longer context makes cached decoding faster. Single-run
  variability and hardware execution effects were not isolated.
- Each new query still attends to a growing set of keys and values. Reusing
  K/V avoids recomputation; it does not remove context-dependent attention work.
- Do not substitute shell command durations for the internal timers: loading,
  terminal output, and menu interaction are not inference-only measurements.
- The output limit alone does not guarantee equal generated counts: EOS may
  stop another workload early. Always record actual counts.
- The learner declined medians/repeated benchmark trials. Preserve the
  single-run qualification rather than claiming statistical certainty.

## Experiment 3: KV-cache memory

### Question and prediction

What is the memory price of retaining K/V, and how does it grow with context?
The learner predicted that doubling cached sequence length doubles K/V storage,
with model, batch size, and dtype fixed.

### What is stored

`model.model.layers[0]` is the first Transformer layer, including its weights
and computation. `past_key_values.layers[0]` is that layer's cache object,
holding computed keys and values for processed positions, not weights,
token IDs, probabilities, or completed answers. Every layer has its own cache.

The cached K/V tensors are reshaped into heads; keys have also received rotary
positional embeddings. They differ from the raw projection shapes printed by
`architecture.py`. Queries are computed for new positions and are not retained
in this KV cache.

Observed after long-prompt prefill:

```text
First-layer keys shape:   [1, 2, 516, 64]
First-layer values shape: [1, 2, 516, 64]
Cache dtype:             torch.float32
Bytes per element:       4
Cached layers:           24
```

Axes: `[batch_size, KV_heads, cached_positions, head_dimension]`.
The cache uses 2 KV heads, not the model's 14 query heads.

### Calculation and direct inspection

For this full-context cache:

```text
KV bytes = 2 × layers × batch × KV_heads × positions × head_dimension × bytes/element
Bytes per position = 2 × 24 × 1 × 2 × 64 × 4 = 24,576 bytes = 24 KiB
```

The script directly sums `numel() * element_size()` for both `.keys` and
`.values` in every entry of `past_key_values.layers`. Inspections occur outside
the timed intervals.

| Point | Cached positions | Calculated bytes | Printed logical K/V storage |
|---|---:|---:|---:|
| After prefill | 516 | 12,681,216 | 12.09 MiB |
| After generation | 615 | 15,114,240 | 14.41 MiB |
| Growth | 99 | 2,433,024 | 2.32 MiB |

The measured storage agrees with the formula. Of 100 selected output tokens,
only the first 99 are processed by another forward pass; selection alone does
not grow the cache. Model weights stay fixed while retained K/V grows.

This is logical tensor storage, not total process RSS, allocator overhead,
temporary tensors, or peak allocation. Linear growth was observed here; do not
generalize the formula unchanged to sliding-window, quantized, or differently
allocated caches.

Illustrative serving calculation, not a measured concurrency experiment:
100 independent requests with 615 cached positions each would need about
1.41 GiB of K/V alone at this configuration, without prefix sharing. Weights
can be shared within a model worker while each request has its own attention state.

### Later memory-inspection runs, kept separate

These are additional user-reported runs after adding memory inspection, not
replacements for the original paired timing comparison above.

| Run | Cached prompt forward | Cached subsequent generation | Reported storage |
|---|---:|---:|---|
| First pasted inspection | 499.95 ms | 7224.55 ms | Prefill 12.09 MiB; final length 615, final bytes not shown |
| Complete pasted inspection | 441.97 ms | 6970.56 ms | Prefill 12.09 MiB; final 14.41 MiB |

Both reported 99 tokens after the first and stopped at the token limit.
Do not cherry-pick the fastest timing across these runs or pair one with an
unrelated uncached run as though they were a new controlled comparison.

## Experiment 4: manual generation versus Hugging Face

### What was compared

Reuse the same loaded model and the original `model_inputs`. Match model,
device, dtype, attention mask, EOS IDs, output limit, and greedy settings.
Do not pass the already-grown input sequence or completed cache into the
reference call.

Each script now creates a `GenerationConfig` and calls `model.generate()`
after the manual experiment, outside its timed intervals. For this decoder-only
model, the reference returns prompt plus generated tokens; slice off the
original prompt length before comparing against the manual `generated_ids`.
The comparison prints token counts and exact list equality, then asserts it.

### User-reported correctness results

| Input and stopping case | Manual cached vs cached generate | Manual uncached vs uncached generate |
|---|---|---|
| 516-token prompt, 100 output tokens, token limit | Exact token-ID match | Exact token-ID match |
| 28-token prompt, 10 output tokens including EOS | Exact token-ID match | Exact token-ID match |

The short prompt is `Reply with exactly one word: Hello`. The observed response
was `Hello! How can I assist you today?`. Ignoring the one-word instruction is
model behavior, not a generation-loop mismatch. Both paths stopped on EOS
before the 100-token limit.

For the short cached run: 28 prefill positions, 9 subsequent generated tokens,
and final cache length `28 + 9 = 37`. Printed logical KV storage was 0.66 MiB
after prefill and 0.87 MiB after generation.

### Generation defaults correction

The initial comparison warned that `temperature`, `top_p`, and `top_k` were
invalid for greedy decoding. Inspection of the installed Transformers code
showed that unset `GenerationConfig` fields are filled from the model's saved
configuration. A fresh config alone does not isolate all model defaults.

The learner made these settings explicit in both comparison configs:

```python
do_sample=False
num_beams=1
repetition_penalty=1.0
temperature=1.0
top_p=1.0
top_k=50
```

`top_k=50` is the installed library's default accepted by greedy validation;
it does not restrict greedy selection because sampling filters are inactive
when `do_sample=False`. Each script uses its matching `use_cache` setting,
the model's EOS IDs, tokenizer pad ID, and the same `max_new_tokens`.
Later learner runs showed exact matches without the warning.

These cases establish agreement on the tested inputs and stopping conditions,
not universal equivalence or factual accuracy. Sampling parity is not tested.

## Experiment 5: CPU versus Apple GPU (MPS)

### Setup and timing correction

The learner confirmed `torch.backends.mps.is_built()` and `is_available()`
both returned `True`. MPS is PyTorch's Apple GPU backend, not CUDA.

Use only `src/cached_generation.py` for the device comparison. Keep Qwen 0.5B,
float32, batch size 1, the same 28-token prompt, greedy selection, cache reuse,
EOS rules, and the 100-token limit fixed. Move the loaded model with `.to(device)`;
the existing `.to(model.device)` moves tokenizer outputs to the same device.

Initial MPS execution was correct but lacked explicit synchronization. Its
reported 158.91 ms prefill and 797.98 ms subsequent time are not the final
properly bounded CPU/MPS comparison. GPU work can be queued before completion.

Four conditional `torch.mps.synchronize()` calls were added:

1. Before starting the prefill timer, so warm-up work has finished.
2. Before stopping the prefill timer, so measured prompt work has finished.
3. Before starting the subsequent-generation timer.
4. Before stopping that timer.

These guards run only when `device == "mps"`. They do not introduce an explicit
synchronize after every forward call, although the loop's `next_token.item()`
already requires each selected token to become available to Python.
Synchronization corrects measurement boundaries; it is not a speed optimization.

### Final paired observations reported by the learner

The learner ran MPS first, then CPU, using the same cached script.

| Measurement | CPU float32 | MPS float32 |
|---|---:|---:|
| Prompt tokens | 28 | 28 |
| Generated tokens including EOS | 10 | 10 |
| Prompt forward pass | 80.29 ms | 43.95 ms |
| Subsequent generation, 9 tokens | 669.87 ms | 246.99 ms |
| Average subsequent time/token | 74.43 ms | 27.44 ms |
| Final cache length | 37 | 37 |
| Final logical K/V storage | 0.87 MiB | 0.87 MiB |
| Manual vs generate on that device | Exact match | Exact match |

Observed MPS speedup: `80.29 / 43.95 = 1.83x` for prefill and
`669.87 / 246.99 = 2.71x` for subsequent generation.

The printed responses matched across devices, and the exact token-ID assertion
passed against `generate()` on each device. No separate cross-device ID
comparison was recorded. Equal logical K/V bytes reflect equal shapes and dtype,
not necessarily equal total memory allocations on the two backends.

This is one short single-request observation per device, not a steady-state
benchmark or a guarantee for other prompts. Only prompt warm-up is performed;
first-use decode initialization is not separately excluded. Do not mix these
measurements with the earlier 516-token runs.

## Sampling scope and planned HTTP behavior

The learner said they had already explored temperature and explicitly chose to
skip the manual top-k/top-p lessons. Mark these skipped, not implemented.

- The manual cached loop remains greedy only.
- The manual uncached loop supports greedy or temperature-based multinomial
  sampling, but neither manual loop implements top-k/top-p filtering.
- Proposed server path: keep these scripts as educational references and use
  Hugging Face `generate()` for the first HTTP implementation.
- A client can request supported sampling settings, but the server must validate
  them and pass them to a runtime that actually implements them. Do not accept
  parameters and silently ignore them.
- The planned initial API includes `top_p`; top-k can remain an explicit internal
  setting rather than a public request field. Sampling filters are inactive for
  greedy decoding. Define and document parameter combinations accordingly.
- Make generation defaults explicit instead of inheriting Qwen's saved defaults.
  Skipping manual filter implementation does not prevent using library sampling.

## Repository checkpoints and documentation

Recent commits inspected while updating this log:

| Commit | Date | Checkpoint |
|---|---|---|
| `e18ed80` | 2026-10-12 | Cached greedy generation, uncached script rename, script documentation |
| `11ccaba` | 2026-10-13 | Cached/uncached latency measurement and documentation |
| `c75022f` | 2026-10-13 | KV-cache timing results and session checkpoint |
| `1e4c4ee` | 2026-10-13 | Prompt-length experiments, KV-memory inspection, and tracked learning log |

The current files additionally include `generate()` parity checks and the cached
script's CPU/MPS selection and synchronized timing. These are pending changes
alongside this log update at the time of inspection. No commit was created by
the agent as part of this update.

Documentation roles: README is the entry point and run guide; this file holds
experiment evidence and the live learning handoff; `plan.md` is the roadmap.
Keep detailed measurements here rather than copying them into every document.

## Exact stopping point and how to resume

Generation parity checks now pass for both the long token-limit case and the
short early-EOS case. The first synchronized CPU/MPS comparison is complete:
MPS showed 1.83x prefill and 2.71x subsequent-generation speedup in the recorded
short-prompt runs. Do not restart these exercises or force the skipped sampling
lessons before continuing.

Stage 2 Checkpoint 1 is complete: the learner distinguished shared model weights
from request-local KV caches, client-supplied conversation history from retained
server state, and immediate busy rejection from a future request queue.

With permission, implemented the loading portion of Checkpoint 2 in
`src/inference.py`: `Runtime` and `load_runtime()`. Loading is explicit, uses
CPU float32 and evaluation mode, preserves configured EOS IDs, and validates
the model context limit and availability of a pad/EOS token. The caller owns
the returned runtime and must reuse it; the loader is not a singleton.

Verification on 2026-10-15 using `.venv/bin/python`:
- Import succeeded while both `from_pretrained` entry points were patched to
  raise if invoked, proving import does not load model/tokenizer resources.
- An explicit real Qwen load returned device `cpu`, dtype `torch.float32`,
  training mode `False`, EOS IDs `[151643, 151645]`, pad ID `151643`, declared
  context `32768`, and supported context `2048`.
- Throwaway configuration checks passed for single/absent EOS normalization,
  EOS fallback eligibility, a smaller declared context, rejection of six invalid
  context limits (`None`, `True`, `0`, `-1`, `2048.0`, `"2048"`), and rejection
  when both pad and EOS are absent.

Completed the reusable generation portion of Checkpoint 2 on 2026-10-16:
`GenerationResult`, `ContextLimitError`, and `generate()` now live in
`src/inference.py`. The function formats client messages without adding an
application system message, tokenizes without duplicate special tokens or
truncation, checks the combined budget, builds a fresh generation config, and
returns completion-only text, prompt count, generated IDs, and finish reason.
Inference mode is entered inside the function, including when called in a
worker thread. Optional preparation and streamer hooks are available for the
later HTTP implementation.

Real Qwen smoke evidence, using one CPU runtime and `.venv/bin/python`:

| Case | Settings | Prompt tokens | Completion tokens | Finish |
|---|---|---|---|---|
| First independent request | Greedy, cap 16, top-p 1 | 36 | 10 | `stop` |
| Second independent request | Greedy, cap 16, top-p 1 | 43 | 2 | `stop` |
| Sampling | Temperature 0.7, top-p 0.9, cap 16 | 36 | 10 | `stop` |
| Small output cap | Greedy, cap 1, top-p 1 | 43 | 1 | `length` |
| Exact combined budget | Greedy, cap 28, top-p 1 | 2020 | 23 | `stop` |

- The two requests asked for a greeting and the answer to `2 + 2`. The arithmetic
  completion decoded to `4`; its second generated ID was EOS. With cap 1, the
  same visible answer had finish reason `length`, demonstrating that visible
  text alone does not determine usage or finish reason.
- Repeating the first greedy request produced identical generated IDs.
- Calls reused the runtime with model/tokenizer loading disabled after startup.
- Real worker forward calls observed inference mode enabled. The saved model
  generation config was unchanged, and no invalid-generation-flag warnings
  appeared.
- Library-streamed text matched ordinary decoded text. This was not an HTTP or
  incremental-delivery test.
- The actual formatted boundary prompt used 2020 tokens: cap 28 was accepted;
  cap 29 raised `ContextLimitError` before generation or preparation callback.
- Temperature zero with top-p 0.9 was rejected before generation.
- Throwaway fixtures verified EOS exactly at the cap reports `stop` and
  unexpected generation failures propagate. Import isolation also passed again.

Implemented the Python validation portion of Checkpoint 3 in `src/api.py`:
`ChatMessage` and `ChatCompletionRequest`. Added the direct dependency
`pydantic>=2,<3`; the existing environment installed Pydantic 2.14.1 without
replacing PyTorch or Transformers. FastAPI and Uvicorn remain for the lifecycle
checkpoint.

The models enforce strict numeric/boolean types, finite numeric ranges, a fixed
model ID, supported roles, nonempty messages/content, and forbidden extra fields.
Whitespace-only content is rejected while valid text is preserved exactly.
Consecutive same-role messages are accepted. Integer temperature/top-p values
are accepted, but booleans and numeric strings are not.

Cross-field validation exposes `invalid_parameter_combination` for greedy
decoding with non-default top-p. Streaming is temporarily rejected with
`unsupported_parameter`. These are Pydantic validation codes, not observed HTTP
responses; no endpoint or HTTP 422 mapping exists yet.

Verification:
- `.venv/bin/python -m unittest discover -s tests -v`: 11 passing behavioral
  tests in `tests/test_api.py`, covering Python and JSON validation, numeric
  boundaries/types, nonfinite numbers, malformed JSON, unsupported fields/model/
  roles, content preservation, incompatible settings, and streaming rejection.
- A separate executable smoke accepted a valid request and filled defaults
  (100 output tokens, temperature 0, top-p 1, stream false). It rejected numeric
  strings, numeric booleans, unknown fields, incompatible settings, and streaming
  with the expected distinct validation types.
- Import and validation succeeded with model/tokenizer loading entry points
  patched to fail if called. No Qwen weights were loaded for these checks.
- A private marker used as an invalid numeric value was absent from the rendered
  exception string. `hide_input_in_errors=True` does not sanitize structured
  `.errors()` results; the future HTTP boundary still needs deliberate error
  mapping without raw inputs or arbitrary exception text.

Verified Checkpoint 4's inference-side mapping on 2026-10-17 without changing
production code. The existing implementation already supplies explicit
generation settings. The cached Qwen generation config still declares sampling
enabled, temperature 0.7, top-p 0.8, top-k 20, repetition penalty 1.1, pad ID
151643, and EOS IDs `[151645, 151643]`; no new saved sampling modifiers were
present.

A throwaway `.venv/bin/python` harness loaded Qwen once and used
`ChatCompletionRequest.model_validate_json()` before calling `generate()`.
It passed `request.max_tokens` as `max_new_tokens`, converted each validated
message with `model_dump()`, and forwarded temperature/top-p explicitly.
The common prompt was `What is 2 + 2? Reply with only the number.`

| Case | Temperature / top-p | Requested cap | Prompt tokens | Generated tokens | Finish |
|---|---|---|---|---|---|
| Greedy before sampling | 0 / 1 | 16 | 43 | 2 | `stop` |
| Sampling | 0.7 / 0.9 | 16 | 43 | 2 | `stop` |
| Greedy after sampling | 0 / 1 | 16 | 43 | 2 | `stop` |
| Small output cap | 0 / 1 | 1 | 43 | 1 | `length` |
| Missing tokenizer pad ID | 0 / 1 | 16 | 43 | 2 | `stop` |

All five completions decoded to `4`. Equal sampled and greedy text is valid,
not evidence that sampling was disabled. The harness observed the real
generation-call settings: sampling used `do_sample=True`, temperature 0.7,
top-p 0.9, and top-k 0; greedy used `do_sample=False` and library values
temperature 1, top-p 1, and top-k 50. Top-k 50 is inactive in greedy mode.

Additional observations:
- Greedy IDs were identical before and after sampling.
- Every call received a distinct `GenerationConfig`; cache enabled, one beam,
  one returned sequence, and repetition penalty 1 were explicitly set.
- The saved model generation config remained unchanged after each call.
- Both configured EOS IDs were preserved. Ordinary pad ID was 151643.
- Temporarily removing the tokenizer pad ID selected the first configured EOS
  ID, 151645, and real generation succeeded. The tokenizer pad ID was restored
  in `finally`; no pretrained file was modified.
- Temperature 0/top-p 0.9 was rejected by the request model with
  `invalid_parameter_combination` and by direct inference with `ValueError`,
  before any model-generation call.
- The exercise recorded zero Python warnings and zero Transformers warnings.

To exercise the same request-to-inference boundary from the repository root:

```bash
.venv/bin/python - <<'PY'
from src.api import ChatCompletionRequest
from src.inference import MODEL_ID, generate, load_runtime

runtime = load_runtime()
for temperature, top_p in [(0, 1), (0.7, 0.9), (0, 1)]:
    request = ChatCompletionRequest.model_validate({
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": "What is 2 + 2? Reply with only the number."}
        ],
        "max_tokens": 16,
        "temperature": temperature,
        "top_p": top_p,
    })
    result = generate(
        runtime,
        [message.model_dump() for message in request.messages],
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    print(temperature, top_p, result.text, result.generated_ids, result.finish_reason)
PY
```

The inference-side checkpoint is verified. Production HTTP binding of
`max_tokens` and the HTTP 422 error response remain pending; this exercise was
Python-to-model, not HTTP. No extra wrapper, permanent test, or endpoint was
added for these verification-only exercises.

Verified the inference portion of Checkpoint 5 on 2026-10-17. The production
budget check already met the requirements, so `src/inference.py` was unchanged.
Added seven behavioral regressions in `tests/test_inference.py`:
- Exact-fit acceptance counts every history turn and template marker and retains
  the full requested output allowance.
- One-token-over rejection occurs before the preparation callback and model
  generation; the error includes counts without private content.
- An oversized prompt is rejected rather than silently shortened.
- Earlier turns cannot be discarded to make a request fit.
- Valid preparation signals precede actual model execution.
- The runtime respects a smaller declared model context and the local 2048 cap.
- Missing, noninteger, boolean, or nonpositive declared limits fail loading.

The boundary fixtures use an in-memory tokenizer and a tiny locally constructed
GPT-2 model, not downloaded Qwen weights. Generation is real for accepted test
inputs, with EOS disabled to exercise the entire output budget. Initialization
uses an isolated random state. Separate loading-error fixtures do not load
weights. This keeps the regression suite deterministic and offline.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result: all 18 tests passed (11 request-schema tests and 7 new inference tests).

Separate real-Qwen smoke:
- Loaded one CPU runtime and supplied four messages: a system instruction, a
  user message containing `"hello " * 1900`, an assistant reply, and a final
  user request for the number 4.
- Message content alone used 1920 tokens. Full chat formatting used 1943,
  including 23 tokens of template overhead.
- Requested output cap 105 exactly fit `1943 + 105 = 2048`. Actual generation
  ended at 19 output tokens with finish reason `stop`.
- Observed generation inputs matched the entire independently tokenized chat,
  and the configured output cap remained 105.
- The same prompt with cap 106 passed request-schema validation but raised
  `ContextLimitError`: prompt count 1943, requested output 106, limit 2048.
  Neither the preparation callback nor model generation ran.
- Repeated that rejection with a `TextIteratorStreamer` supplied directly to
  inference. It emitted no output and again did not signal preparation or run
  the model. This is library-level coverage, not an accepted HTTP stream request.

The first smoke harness incorrectly assumed `apply_chat_template(tokenize=True)`
returned a token list. Inspection showed the installed Transformers returns a
`BatchEncoding`; reading its `input_ids` corrected the harness, and the smoke
then passed. Production already counts tensor input IDs and was unaffected.

HTTP 400 mapping and pre-SSE-header rejection remain unverified because no
chat-completion endpoint exists. The plan keeps these items unchecked instead
of treating a Python exception as an HTTP response.

Before Checkpoint 6, repaired a user-reported malformed section in
`tests/test_api.py`: the numeric-range loop was incomplete and the nonfinite,
streaming, and greedy/top-p test definitions were missing. Restored the earlier
behavioral cases without changing production validation. Running
`.venv/bin/python -m unittest discover -s tests -v` afterward passed all 18 tests.

Implemented Checkpoint 6 in `src/api.py`: a FastAPI lifespan owns the runtime,
single-worker executor, admission guard, and initialized/shutdown flags.
Successful startup loads Qwen once; loading or executor-creation failure aborts
startup and cleans up owned references. Shutdown marks unavailable, waits for
executor work via `asyncio.to_thread`, then clears executor/runtime references.
The existing inference implementation and request validators are unchanged.

Added FastAPI and Uvicorn to `requirements.txt`, and declared `httpx` for HTTP
tests (already installed). Installed FastAPI 0.143.3 and Uvicorn 0.54.0.
Before/after version checks confirmed PyTorch 2.14.1, Transformers 5.18.0, and
Pydantic 2.14.1 were unchanged.

Six new lifecycle regressions in `tests/test_api.py` verify:
- Repeated health/readiness requests share one loaded runtime and normal
  shutdown clears resources and closes the executor.
- Loading failure prevents entry into the serving phase.
- Executor-creation failure cleans up the already-loaded runtime.
- An occupied guard produces 503 busy while health remains 200; releasing it
  restores readiness.
- Uninitialized state returns 503 not_ready.
- Shutdown waits for a deliberately held producer without blocking the HTTP
  event loop. While waiting, health returns 200, readiness returns 503
  shutting_down, and the runtime remains referenced through producer exit.

`.venv/bin/python -m unittest discover -s tests -v` passed all 24 tests.
The lifecycle fixtures do not load Qwen; shutdown-time responses are controlled
ASGI observations, not claims about reachability during real Uvicorn shutdown.
A separate fresh-process import smoke blocked both model/tokenizer loading
entry points and successfully imported `src.api`.

Real localhost smoke used:

```bash
.venv/bin/python -m uvicorn src.api:app \
  --host 127.0.0.1 --port 8000 --workers 1
curl --fail --silent --show-error --include \
  http://127.0.0.1:8000/health http://127.0.0.1:8000/ready \
  http://127.0.0.1:8000/health http://127.0.0.1:8000/ready
```

Observed one Qwen weight-loading progress sequence before application startup
completed. Both health responses were 200 with `{"status":"alive"}`; both ready
responses were 200 with `{"ready":true,"status":"ready"}`. No additional loading
occurred during those requests. Ctrl-C produced application shutdown complete
and process exit code 0; the smoke server is stopped.

No pre-start HTTP readiness response was collected. No chat-completion route,
production admission path, generation request IDs, or SSE was added. The busy
and shutdown states were verified with controlled fixtures, not concurrent
real-model HTTP generation. README now documents launch, readiness, shutdown,
and test commands.

Implemented the internal Checkpoint 7 execution path in `src/api.py`:
- `submit_inference(app, request)` accepts a validated request on the event loop,
  rejects unavailable state or an occupied guard without waiting, and submits
  only admitted work to the dedicated executor.
- `InferenceUnavailable` and `InferenceBusy` distinguish admission failures.
  Submission failures release the guard and propagate unchanged.
- `_run_inference()` performs message conversion and reusable generation on the
  worker, including tokenization and context validation. Its `finally` owns
  guard release.
- `_observe_completion()` removes completed futures and retrieves exceptions
  without logging raw details. Both the underlying producer and returned
  consumer result are observed.
- `app.state.producer_futures` retains underlying work. The returned future is
  already shielded, rather than requiring every caller to remember a shield.
  Canceling it does not cancel the producer or release the guard. Adding another
  caller-side shield is unnecessary but does not leave abandoned errors
  unobserved.
- Lifespan shutdown waits for executor work and drains retained future
  completion before clearing runtime references.

Added five admission tests and updated the held-worker shutdown regression to
use actual admission. Covered busy rejection without queueing, off-event-loop
execution, consumer cancellation while busy, context/generation failure
recovery, submission failure cleanup, unavailable-state rejection, and failure
observation after the consumer abandons its result. A loop exception-handler
check verifies abandoned failures do not become unhandled-future errors.
Shutdown also rejects new submissions and retains runtime through producer exit.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Result: all 29 tests passed. No new dependency or inference-algorithm change was
needed.

A temporary real-Qwen harness started the actual Uvicorn application on an
ephemeral `127.0.0.1` port and submitted validated requests on its event loop.
Generation settings were greedy, top-p 1, output cap 16. The prompt was
`What is 2 + 2? Reply with only the number.`

Observed:
- One runtime load across all requests.
- Initial HTTP readiness 200.
- After one successful real forward call, a controlled gate held the worker.
  HTTP health remained 200 and readiness returned 503 busy. A second internal
  submission immediately raised `InferenceBusy`; this was not HTTP 429.
- Canceling the Python result future left the producer pending and retained.
  Another submission still raised `InferenceBusy` until the gate was released.
- Actual generation finished with 43 prompt tokens, 2 completion tokens, and
  `stop`; HTTP readiness returned 200.
- A 2530-token formatted prompt with output cap 16 raised `ContextLimitError`
  against the 2048 limit and released admission.
- A temporary wrapper raised on the second real `model.forward` call. The
  exception propagated and admission recovered. The original method was
  restored by the patch context.
- A subsequent real request decoded to `4`, with 43 prompt tokens,
  2 completion tokens, and `stop`. Readiness was 200 and no producer futures
  remained retained.
- Uvicorn shut down cleanly; executor/runtime references were cleared and the
  harness exited with code 0. No harness file or server remains.

The real HTTP traffic exercised health/readiness only. Generation was submitted
internally; no completion route, HTTP 429/503/500 error mapping, SSE preparation
signal, or network-disconnect verification was added. The plan keeps those
acceptance items pending.

Implemented Checkpoint 8 on 2026-10-18: `POST /v1/chat/completions` in
`src/api.py` now accepts the validated request model, awaits the existing
shielded admission path, and returns ordinary JSON with assistant content,
completion ID, integer creation timestamp, fixed model ID, finish reason, and
actual token usage. No inference algorithm or dependency changes were needed.

Each handled chat request receives a server-generated UUID in `X-Request-ID`;
successful completion IDs use the `chatcmpl-` prefix and errors include the same
request ID as the header. Incoming identifiers are ignored. Structured lifecycle
logging remains Checkpoint 11; request IDs were needed now for the error contract.

Mapped errors:
- Schema/JSON failures: 422 `invalid_request`.
- Greedy/top-p mismatch: 422 `invalid_parameter_combination`.
- Streaming requested before implementation: 422 `unsupported_parameter`.
- Context overflow: 400 `context_length_exceeded`, including counts but no input.
- Occupied producer: 429 `server_busy`.
- Unavailable lifecycle state: 503 `not_ready`.
- Submission or unexpected inference failure: 500 `generation_failed`, with no
  raw exception text or traceback.

Validation messages use fixed, known descriptions rather than serializing raw
Pydantic inputs, contexts, or unknown field names. A regression initially exposed
FastAPI returning 400 for malformed JSON encoding; the boundary now normalizes
that Unicode parsing failure to the documented 422 response. Other framework
HTTP exceptions retain default handling, including 404/405. One test fixture
also needed a real whitespace escape instead of literal backslash-n.

Added six endpoint regressions for sanitized validation before busy admission,
unavailable state, context/generation error recovery, submission failure cleanup,
busy rejection and canceled HTTP-handler ownership, and preserved routing.
`.venv/bin/python -m unittest discover -s tests -v` passed all 35 tests after
those fixes. Handler cancellation was exercised through ASGI, not a real
network disconnect.

A temporary harness ran the actual Uvicorn app on an ephemeral localhost port,
loaded Qwen once, and used real curl subprocesses to send HTTP requests.
The ordinary prompt was `What is 2 + 2? Reply with only the number.`

| Case | HTTP | Prompt / completion / total tokens | Finish |
|---|---|---|---|
| Greedy, cap 16 | 200 | 43 / 2 / 45 | `stop` |
| Repeated greedy, cap 16 | 200 | 43 / 2 / 45 | `stop` |
| Greedy, cap 1 | 200 | 43 / 1 / 44 | `length` |
| Sampling, temperature 0.7, top-p 0.9, cap 16 | 200 | 43 / 2 / 45 | `stop` |
| Exact context fit, cap 28 | 200 | 2020 / 23 / 2043 | `stop` |
| Post-failure recovery, greedy cap 16 | 200 | 43 / 2 / 45 | `stop` |

The exact-fit request reserved `2020 + 28 = 2048`, then stopped early after
23 generated IDs; reported usage did not falsely claim the full cap. The same
prompt with cap 29 returned 400 `context_length_exceeded`. A larger prompt
returned the same code with counts 5030 + 16 > 2048.

Additional real-HTTP observations:
- Unsupported model, empty messages, negative cap, unknown private field,
  nonfinite temperature, malformed JSON, and invalid JSON encoding returned 422.
- Temperature 0/top-p 0.9 and stream true returned their distinct 422 codes.
- A controlled gate after a successful Qwen forward call held an active HTTP
  completion. Health returned 200; readiness returned 503 busy; a second valid
  completion returned 429. Invalid input still returned 422 while busy.
- Releasing the gate let the original curl request finish with 200.
- A controlled exception on the second real forward call returned sanitized
  HTTP 500. Restoring the method allowed the next real completion to succeed.
- A temporarily forced shutting-down state returned HTTP 503 `not_ready`; this
  was controlled state, not a claim about connectivity during real shutdown.
- GET on the completion route returned 405; an unknown route returned 404.
- Response IDs were distinct, server-generated, and correlated with error
  headers. Usage and finish metadata matched the actual inference results.
- A private marker in invalid values, unknown field names, the incoming request
  ID, and the injected exception was absent from error responses and captured
  server logs. This does not constitute the future lifecycle-logging feature.
- Readiness returned to 200, all producer references drained, and the smoke
  process shut down with exit code 0. No temporary harness/server remains.

README now documents the ordinary curl request, supported fields, response
metadata, error codes, and remaining scope. The plan marks previously pending
HTTP acceptance criteria from Checkpoints 3-7 only where now verified.

## Checkpoints 9 and 10: streaming and disconnect ownership

The learner explicitly selected implementing Checkpoints 9 and 10 together.
`src/api.py` now accepts `stream=true` and passes a per-request
`TextIteratorStreamer` through the existing single admitted producer. Its
configuration skips prompt/special tokens, preserves decode spacing, and uses a
0.1-second queue timeout. No inference algorithm or dependency changes were needed.

A preparation future resolves after formatting, tokenization, context validation,
and generation configuration, or with preparation failure. The endpoint waits
before returning SSE. Queue reads run off the event loop and translate text,
timeout, and sentinel into ordinary values. A timeout is not failure; producer
failure terminates delivery even without a sentinel. A sentinel alone is not
success: the consumer waits for the producer result.

Successful SSE sends an assistant-role chunk, nonempty content chunks, an empty
terminal delta with actual finish reason and usage, then exactly one `[DONE]`.
ID, model, and creation time stay consistent. Terminal usage is a documented
local extension; `stream_options` remains unsupported. Counts use generated
IDs, including invisible EOS, rather than chunk count or re-tokenized text.
Known pre-header failures use ordinary HTTP errors. Post-header producer failure
sends sanitized `generation_failed` SSE and `[DONE]` if connected, with no
successful terminal finish reason or fabricated usage.

Both ordinary and streaming handlers now stop delivery after detected disconnect,
while the retained producer alone owns admission until actual exit. A closed
pre-header connection receives no fabricated HTTP response. Abandoned producer
errors remain observed. Finite generation runs under the existing cap rather
than pretending a canceled coroutine stops native model computation.

Seven new streaming/ownership regressions cover:
- Incremental content before producer completion and token-based terminal usage.
- Empty queue timeouts without false failure.
- Preparation failure before headers and later recovery.
- Submission failure without an unresolved preparation waiter.
- Failure both without a sentinel and after a sentinel precedes the result.
- Ordinary/preparation disconnects without sending a response or freeing admission.
- Post-content disconnect ownership under ASGI 2.3 and 2.4 response mechanisms.

The first full run exposed a real cancellation race in the existing ordinary
HTTP test: Starlette's AnyIO disconnect probe could consume concurrent task
cancellation. The probe wrapper now preserves pending `Task.cancel()` state.
After the fix, `.venv/bin/python -m unittest discover -s tests -v` passed all
42 tests. The obsolete streaming-rejection expectations were removed.

Real-Qwen Uvicorn smoke runs used temporary in-memory harnesses, ephemeral
localhost ports, actual `curl -N` subprocesses, HTTPX, and TCP connections.
Each process loaded one CPU float32 runtime. Greedy SSE and ordinary output
matched byte-for-byte for these observed cases:

| Case | Prompt / completion / total tokens | Content chunks | Finish |
|---|---|---|---|
| Chinese text and punctuation | 47 / 5 / 52 | 3 | `stop` |
| Portuguese diacritics and punctuation | 54 / 18 / 72 | 5 | `stop` |
| English list capped at 24 tokens | 43 / 24 / 67 | 17 | `length` |

Every case delivered nonempty content before producer completion. Both EOS
cases ended with token ID 151645, included in usage. The Portuguese response was
`Olá, café! (ação) 1,23? (ação)`; generation parity does not imply perfect
instruction following. The Chinese response was `你好，世界！`. These are
correctness observations, not latency benchmarks.

Streaming context boundaries were also exercised with the real tokenizer:
a 2020-token formatted prompt plus cap 28 exactly fit 2048 and returned
23 actual completion tokens. Cap 29 and oversized input returned HTTP 400 JSON
`context_length_exceeded` before SSE headers.

Real disconnect/failure evidence used controlled gates around actual Qwen
forward calls, without adding a public failure-injection endpoint:
- Ordinary TCP client closed while the producer was held after one successful
  forward call. The HTTP handler exited while admission stayed occupied.
  Releasing the gate raised a controlled model failure, which was observed.
- Streaming TCP client closed after receiving nonempty content. Its handler
  exited while the worker retained admission, then the worker finished its
  24-token cap after release.
- A connected stream failed after its fourth successful forward call. It kept
  HTTP 200, emitted sanitized error SSE plus exactly one `[DONE]`, and emitted
  neither successful terminal metadata nor fabricated usage. No streamer
  sentinel was needed to end the failed response.
- During every held call, health returned 200, readiness returned 503 busy, and
  another valid completion returned 429. After worker exit, readiness returned
  200 and a real recovery request returned `4`.
- All smoke processes exited with code 0 and released runtime, executor, and
  retained producers. The ownership harness observed no unhandled loop failures.
  No temporary scripts, server processes, or failure hooks remain.

README now documents streaming curl usage, frame semantics, the local usage
extension, error boundaries, and disconnect ownership. The plan marks completed
Checkpoints 9-10 and their previously deferred context/admission criteria.

At the end of Checkpoints 9-10, private lifecycle logging was the next task.
Cooperative model stopping, bounded slow-consumer backpressure, and native-call
recovery remain outside this stage. A permanently stuck producer requires restart.

Still not established: a formal temperature study, manual top-k/top-p filtering
(explicitly skipped), repeated statistical benchmarks, total/peak process-memory
profiling, load/concurrency benchmarks, batching experiments, and precision/
quantization comparisons. The whole roadmap is not complete.

## Checkpoint 11: private correlated lifecycle logging

Implemented on 2026-10-21 with the learner's approval. Changes are in
`src/api.py` and new `tests/test_logging.py`; inference algorithms and dependencies
are unchanged. No staging, commits, or pushes were performed for this checkpoint.

The standard-library `inference.lifecycle` logger emits JSON lines to stderr:
- `request_accepted` after successful admission and worker submission.
- Exactly one `request_finished` at final ASGI send completion, detected
  disconnect/cancellation, or terminal failure.
- `producer_finished` separately when admitted worker computation exits,
  even after the HTTP handler has ended.

Pure ASGI middleware assigns the server-generated UUID and starts monotonic
timing before body validation. It observes send/receive boundaries without
buffering bodies and returns the same ID in headers, completion IDs, and errors.
Handlers no longer independently generate request IDs.

Records include only selected metadata: request ID, configured model ID, valid
stream mode, known HTTP status, outcome, duration, actual available token counts,
sanitized error code, and unexpected exception class. Unknown stream/status is
null; unavailable counts are omitted. Arbitrary invalid values, private headers,
messages, outputs, token IDs, and raw exception text are excluded. Unexpected
response errors before headers receive a sanitized HTTP 500; errors escaping
after headers are sanitized before propagation to the server logger.

`duration_ms` belongs to request delivery. `producer_duration_ms` measures the
worker from preparation through generation/output extraction, not network
delivery or executor handoff. A request can end as disconnected while its worker
later reports completed or failed. A failed stream can retain HTTP 200 while its
logged outcome is failed. Successful ASGI send completion is not proof of receipt
by the remote client.

Ten permanent regressions cover:
- Correlation and actual counts without prompt/output/token-ID/header leakage.
- Timing before body validation and after the final ordinary or streaming send.
- Private malformed/schema input and unique correlated rejection records.
- Busy and unavailable rejection without nonexistent producer events.
- Context/generation failures and recovery without stale request metadata.
- Submission failure without false acceptance.
- Post-header failure logged as failed despite HTTP 200.
- Real ASGI disconnect mechanisms and task cancellation before worker exit,
  with distinct delivery/producer timing and outcomes.
- Failed send recorded once as disconnected, not successful.
- Unexpected ordinary response failure sanitized at the application boundary.

`.venv/bin/python -m unittest discover -s tests -v` passed all 52 tests.

A temporary in-memory harness then served the actual Uvicorn application over
an ephemeral localhost port with one CPU float32 Qwen load. Actual curl requests,
HTTPX, and TCP sockets exercised 18 chat requests and produced 36 JSON lifecycle
records. The default logger remained active; captured stdout/stderr included
normal Uvicorn logs and the application's JSON lines.

| Case | Observed behavior |
|---|---|
| Ordinary and SSE success | HTTP 200; matching greedy output; both logged 49 prompt and 14 completion tokens |
| Five private invalid requests | HTTP 422; one correlated rejected event each; no acceptance or producer event |
| Streaming context overflow | HTTP 400 before SSE; producer and request outcomes rejected |
| Controlled unavailable state | HTTP 503; rejected without a producer event |
| Ordinary TCP disconnect | Request ended before headers; held worker later failed with safe `RuntimeError` metadata |
| Streaming TCP disconnect | Request ended after content; held worker later completed 24 tokens from a 40-token prompt |
| Connected forward failure | HTTP 200 stream ended with error SSE and `[DONE]`; request and producer outcomes failed |
| Overlapping requests | HTTP 429 during each held worker; health 200 and readiness 503 busy |
| Three recovery requests | HTTP 200 afterward; 43 prompt and 2 completion tokens; no stale error fields |

The streaming disconnect recorded delivery duration about 331.23 ms, then later
recorded a separate producer duration about 1855.49 ms. These intervals have
different starting boundaries and are single-run correctness observations,
not speed comparisons or TTFT/TPOT measurements.

The private marker supplied in prompt, invalid values/field names, incoming
request ID, authorization header, and injected exception was absent from complete
captured server stdout/stderr. Generated output was absent from lifecycle
records. Every chat request had one delivery terminal event. The ownership
harness observed no unhandled loop failures, retained producers, or leaked
runtime/executor after shutdown; process exit code was 0. Temporary forward
wrappers were restored, and no smoke files or server processes remain.

README documents event semantics, null/omitted fields, privacy, and timing.
Checkpoint 11 is complete. Next: Checkpoint 12, consolidate the final service
acceptance checks and answer the learning questions. Do not mark the entire
phase complete solely from this logging checkpoint.

## Checkpoint 12: complete service verification

Technical acceptance is verified. The learner still needs to answer the
[learning questions](../plan.md#learning-checks); do not mark those complete
from test or smoke results.

### Regression coverage

Reviewed the existing schema, context, lifecycle, admission, ordinary-response,
streaming, disconnect, failure-recovery, and logging regressions. Added the
missing explicit EOS-at-output-cap boundary in `tests/test_inference.py`.
The existing tiny offline model executes real Hugging Face generation with
controlled logits: one visible token followed by EOS at cap 2. Both configured
EOS IDs are exercised separately. The result retains both generated IDs,
decodes only the visible token, and reports `stop`, not `length`.

Full suite:

```text
.venv/bin/python -m unittest discover -s tests -v
Ran 53 tests in 1.172s
OK
```

A separate fresh-process import check made both Hugging Face `from_pretrained`
loaders raise if called. Importing `src.api` succeeded without calling either:
application import does not load weights.

### Real-Qwen HTTP verification

Ran the actual application with Qwen/Qwen2.5-0.5B-Instruct, CPU float32, on an
OS-assigned localhost port in a separate process. Used HTTPX, `curl -N`, and
raw TCP clients for disconnects. The existing user server was not changed.
The runtime loaded once and was reused across 64 chat requests. Real model
forward calls ran in evaluation mode with `torch.inference_mode()` enabled.

| Check | Observed result |
| --- | --- |
| Ordinary greedy arithmetic request | HTTP 200; 43 prompt + 2 completion = 45 tokens; `stop` |
| Same request with output cap 2 | Ordinary and streaming both retained the same generated IDs and reported `stop`; EOS counted despite being invisible |
| Same request with output cap 1 | One generated ID; `length` |
| Sampling at temperature 0.7, top-p 0.9 | HTTP 200; 43 + 2 = 45 tokens; actual generation config used sampling and `top_k=0` |
| Configuration isolation | Per-request generation configs preserved both model EOS IDs, one beam/return sequence, cache use, and repetition penalty 1; saved model configuration remained unchanged |
| Invalid HTTP requests | 28 schema/body cases returned 422 without model execution or private-value echoes; included strict types/ranges, unknown fields, malformed JSON/UTF-8, NaN, and greedy/top-p incompatibility |
| Routing | GET on the chat route returned 405; unknown route returned 404 |
| Exact formatted context budget | 2020 prompt + requested cap 28 = 2048 accepted in both modes; actual usage 2020 + 23 = 2043 |
| One token over | 2020 + requested cap 29 = 2049 returned JSON 400 before model execution in both modes |
| Oversized prompt | Repeated-word prompt returned JSON 400 before model execution in both modes; no truncation or streaming headers |

Sampling is validated by the actual generation settings and successful execution,
not by requiring a different answer from greedy decoding.

For three greedy prompt pairs, ordinary and streaming runs produced identical
generated ID sequences and decoded text. Each actual `curl -N` client received
nonempty content before its producer exited:

| Prompt case | Prompt / completion / total tokens | SSE chunks including role and terminal, excluding `[DONE]` | Finish |
| --- | --- | --- | --- |
| English animal list, cap 24 | 38 / 24 / 62 | 16 | `length` |
| Chinese text with punctuation, cap 64 | 47 / 5 / 52 | 5 | `stop` |
| Portuguese accents and punctuation, cap 64 | 54 / 18 / 72 | 7 | `stop` |

Each successful stream had one role event, stable identity/creation metadata,
one successful terminal event with usage matching actual generated IDs, and
exactly one `[DONE]`. Chunk counts are not token counts. These are correctness
observations, not timing benchmarks or first-token latency measurements.

### Ownership, failures, and shutdown

Temporary wrappers held execution only after a real model forward call; stream
cases waited until real nonempty text had been emitted. While each producer was
held, `/health` returned 200, `/ready` returned 503 with `busy`, valid overlapping
chat requests returned 429, and invalid requests still returned 422.

- Ordinary client disconnect followed by an injected worker failure: delivery
  ended as `disconnected`, but admission stayed busy until producer exit.
  The abandoned `RuntimeError` was observed, readiness recovered, and a later
  real-model request returned 200.
- Streaming client disconnect: delivery ended while the real producer continued.
  Admission stayed busy until completion, then a later request returned 200.
- Connected streaming failure after content: an injected `RuntimeError` produced
  one sanitized SSE error and one `[DONE]`, without successful finish metadata
  or usage. Delivery was logged as failed despite HTTP 200. A later request
  returned 200.
- Graceful shutdown after an ordinary client disconnected: the real producer
  remained held while Uvicorn entered lifespan shutdown. The runtime and busy
  slot remained owned, initialization/readiness state was cleared, new internal
  admission raised `InferenceUnavailable`, and the event loop stayed responsive.
  Uvicorn had closed its listener, so this is not evidence of an HTTP 503 response
  during shutdown. After releasing the worker, shutdown waited for completion,
  then cleared runtime/executor references and retained futures; the guard was
  unlocked.

The controlled gates/errors expose lifecycle transitions; they do not replace
model computation. Existing permanent regressions additionally cover startup
failure, unavailable HTTP 503, preparation/submission failures, and both relevant
ASGI disconnect paths.

### Lifecycle/privacy evidence and remaining work

The run captured 114 actual lifecycle records for 64 chat requests:

- 25 accepted producers: 19 completed, 4 rejected for context, 2 injected failures.
- Delivery outcomes: 17 completed, 43 rejected, 3 disconnected, 1 failed.
- Delivery statuses: 19 HTTP 200, 32 HTTP 422, 4 HTTP 400, 7 HTTP 429, and 2
  disconnected ordinary requests without committed response headers.

Every chat request had exactly one delivery terminal event. Every accepted
request had a corresponding producer terminal event. Private prompt/validation/
exception markers were absent from captured server stderr, and generated private
output was absent from lifecycle records. No raw traceback, invalid-generation-
flags warning, or unhandled async exception was observed.

The smoke process exited successfully after graceful shutdown. Temporary
instrumentation stayed in memory; no smoke files or server processes remain.
No production source changes were needed. README and the Checkpoint 12 technical
checklist now reflect the verified state. Generated changelogs and `.ua` files
were not edited, and no staging, commits, or other Git operations were performed.

Remaining: the learner answers the 15 Checkpoint 12 questions in `plan.md`.
Technical checks are complete; learning verification is intentionally pending.

## Phase 3: reproducible inference benchmarks

The learner authorized implementation of the seven benchmark steps after the
timing-boundary lesson. Production `src/` code, HTTP request/response fields,
dependencies, and the existing user server were left unchanged.

### Tools and measurement contract

- `benchmarks/workloads.py` creates the fixed synthetic manifest using the actual
  cached tokenizer. Inputs are 59 and 990 formatted tokens, each paired with
  output caps 32 and 128. Exact messages, token counts, template hash, and snapshot
  revision are retained.
- `benchmarks/client.py` is a separate sequential HTTP streaming client. It
  records first nonempty content and receipt of `[DONE]`, actual terminal usage,
  failures, and cancellations. It does not count chunks as tokens or retry errors.
- `benchmarks/timing.py` timestamps actual generated-token callbacks, excludes
  the initial full-prompt callback, verifies callback counts against returned
  generated IDs, and reports TTFT, mean TPOT, and individual inter-token intervals.
- `benchmarks/server.py` opts into timing on the real service without changing
  production startup. Scoped wrappers carry request IDs into the existing worker
  and remain installed until the original shutdown path finishes.
- `benchmarks/report.py` joins durations by request ID, rejects token/revision
  mismatches, excludes warm-ups from headline statistics, and leaves unavailable
  measurements null. It never subtracts absolute timestamps across processes.
- `benchmarks/model.py` provides separate natural/fixed model-side experiments
  and a separate profiler mode. Fixed mode changes only the request-local
  `min_new_tokens`, preserving context/output limits and shared model defaults.

Commands and field definitions are in the README benchmark section. All run
artifacts are under ignored `benchmark-results/phase3/`; result writers refuse
to overwrite existing evidence.

### Baseline and startup

Observed environment, rather than a guessed hardware specification:

- Apple M1 Pro, 8 logical CPUs; reported platform macOS 26.6.2 arm64.
- Python 3.12.13, PyTorch 2.14.1, Transformers 5.18.0, Tokenizers 0.23.2.
- FastAPI 0.143.3, Uvicorn 0.54.0, HTTPX 0.28.1.
- Qwen/Qwen2.5-0.5B-Instruct snapshot
  `7ae557604adf67be50417f59c2c2f167def9a775`, CPU float32.
- One worker, one sequential client, 2048-token combined context limit.
- PyTorch reported 6 intra-op and 8 inter-op threads; no thread tuning was applied.
- HTTP benchmark server launch used `HF_HUB_OFFLINE=1` and
  `TOKENIZERS_PARALLELISM=false`, also retained in `server-config.json`.

This Transformers version does not retain `_commit_hash` on its configuration
object. The benchmark therefore discovers the cached snapshot explicitly and
pins both model and tokenizer loading to that revision. No weights were downloaded.
Download time is unmeasured, not zero. The server loaded once: 2433.35 ms for
cached runtime loading and 2516.07 ms from lifespan entry to instrumentation
readiness. The startup interval excludes Python/module imports and all request
measurements.

### Repeated natural-stopping HTTP workload

Executed the actual benchmark client against an isolated localhost instrumented
server. Each case ran two excluded warm-ups followed by 20 measured requests,
in manifest order. All 88 requests succeeded; all 80 measured requests joined
to model records by request ID, with no missing records, errors, or cancellations.
Every first nonempty content delivery preceded model-generation completion.

| Formatted input / output cap | Measured requests | E2E p50 / p95 ms | Client first-text p50 ms | Model TTFT p50 ms | Mean TPOT p50 ms | Useful output tokens/s | Process peak RSS MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 59 / 32 | 20 | 2395.34 / 2505.33 | 189.80 | 115.72 | 73.49 | 13.06 | 1994.03 |
| 59 / 128 | 20 | 10204.78 / 13064.56 | 200.70 | 122.31 | 78.97 | 12.07 | 1994.03 |
| 990 / 32 | 20 | 3303.79 / 3757.35 | 752.22 | 660.09 | 85.28 | 9.55 | 2103.45 |
| 990 / 128 | 20 | 11333.54 / 13829.09 | 807.89 | 719.91 | 82.64 | 10.79 | 2105.03 |

All natural requests reached their caps and reported `length`, producing 6400
measured output tokens. Natural stopping was enabled, but this particular
workload did not demonstrate early EOS. No output text or token IDs were retained
in the per-request measurements.

Pooled inter-token p50/p95 intervals were respectively 72.60/82.50,
74.65/105.79, 79.82/111.71, and 78.88/112.17 ms. These pooled intervals are
different statistics from the distribution of per-request mean TPOT.

For each correlated request, subtracting model-call duration from client E2E
duration left median residuals of 2.43, 5.90, 4.66, and 12.38 ms across the four
cases. The model-generation region occupied a median 99.86-99.95% of E2E time.
That region includes Hugging Face generation logic and streamer callbacks, not
only matrix kernels. This compares paired durations, not cross-process clock
timestamps.

Artifacts: `workload.json`, `server-config.json`, `http.json`, `server.jsonl`,
and `report.json`. The benchmark server exited normally with code 0 before
starting model-side profiling.

### Separate profiler and bottleneck interpretation

Selected `long-input-long-output` because it had the largest measured median
E2E latency. A separate process ran two warm-ups and one profiled natural
generation: 990 input tokens and 128 generated tokens. Its timing is not included
in the HTTP result table.

| Operator | Calls | Self CPU time ms |
| --- | --- | --- |
| `aten::mm` | 12416 | 8093.34 |
| `aten::addmm` | 9216 | 486.60 |
| `aten::_scaled_dot_product_flash_attention_for_cpu` | 3072 | 465.90 |
| `aten::cat` | 12672 | 454.09 |
| `aten::mul` | 28416 | 99.74 |
| `aten::silu` | 3072 | 78.75 |

The profiled generation region lasted 11347.14 ms. Dense matrix multiplication
dominated recorded operator self time; paired client/model durations show that
HTTP delivery overhead was small in this workload. The evidence identifies the
dominant operation, but does not establish whether arithmetic throughput or memory
bandwidth is the hardware limit. Do not claim CPU saturation or a device speedup
without a controlled measurement.

Artifacts: `profile.json` contains samples and operator data, with no headline
case summaries; `profile.trace.json` is the separate 387618436-byte Chrome trace.
Profiler shape/memory collection and trace export add overhead. Operator allocation
totals are not peak process memory, and negative operator memory deltas are not
negative RSS.

### Regression and HTTP timing-boundary evidence

The integrated suite passed: `Ran 73 tests in 1.183s`, `OK`. New regressions cover
SSE truncation/errors, actual usage, first-text versus role events, EOS accounting,
token callbacks, fixed-length isolation, warm-up exclusion, and correlated reports.

A real HTTP fixture exposed an initial client timing error: `[DONE]` arrived
about 2 ms after request start, but HTTP body closure was delayed by 400 ms.
The initial implementation reported about 408 ms. After correcting the boundary,
the same scenario reported about 4 ms despite the closure delay. The permanent
streaming regression now checks that delayed HTTP closure does not inflate E2E.

A separate real HTTP fixture interrupted the client during warm-up. The CLI
returned 130 and saved the canceled sample. Reporting succeeded with zero measured
requests, one warm-up failure, and null throughput/latency rather than invented
zero-valued performance.

### Fixed-length measurements and EOS isolation

After the HTTP server and profiler process had exited, ran the standalone
fixed-length command with two warm-ups and 20 measured generations per case.
All 88 calls succeeded. All 80 measured generations produced exactly their
requested caps, and actual per-call settings recorded
`min_new_tokens == max_new_tokens`.

| Formatted input / output cap | Measured requests | Model-side request p50 ms | TTFT p50 ms | Mean TPOT p50 ms | Output tokens/s |
| --- | --- | --- | --- | --- | --- |
| 59 / 32 | 20 | 2490.98 | 118.19 | 76.55 | 12.74 |
| 59 / 128 | 20 | 9955.30 | 118.22 | 77.47 | 12.77 |
| 990 / 32 | 20 | 3175.44 | 661.26 | 81.08 | 9.77 |
| 990 / 128 | 20 | 10742.39 | 643.83 | 79.50 | 11.62 |

Artifact: `fixed.json`. Model-side request time includes tokenization and final
decoding, but is not HTTP E2E latency. This run has no downstream text streamer
and occurred later in a separate process. Differences from the HTTP table are
not evidence that forcing output length improves serving performance.

Because the four main natural prompts all reached their caps, an additional
real-Qwen correctness control used an arithmetic prompt with 43 formatted input
tokens and an output cap of 8, all on one runtime:

| Mode | Actual generated tokens | Finish reason | Actual `min_new_tokens` |
| --- | --- | --- | --- |
| Natural | 2 | `stop` | unset |
| Fixed | 8 | `length` | 8 |
| Natural again | 2 | `stop` | unset |

The two natural runs produced identical generated ID sequences, and the shared
model generation configuration was unchanged. The terminating EOS was counted
but not printed or saved as an ID. Artifact: `eos-control.json`; this is a
correctness control, not another performance sample.

### Real model failure accounting and recovery

A separate in-memory harness ran the actual instrumented server and raised a
private-marker `RuntimeError` after its fourth successful real forward call.
The timing recorder had observed three generated-token callbacks before failure.
The benchmark client had received nonempty content, then recorded
`generation_failed` with null completion usage rather than counting that partial
output as a completed response.

The same client run then sent another real request, which completed with 32
generated tokens. The CLI saved both samples and exited 1 as required for a run
containing an error. Model/client request IDs matched; the joined report contained
one error, one success, and one known failed producer duration. Only the 32
successfully delivered output tokens contributed to useful throughput, while
the measured elapsed window included the failed request.

Artifacts: `failure-server.jsonl`, `failure-http.json`, and `failure-report.json`.
Readiness recovered to HTTP 200. The harness server shut down normally with code
0; its captured logs contained neither the injected private exception marker
nor a traceback. The injection remained in memory, not in a production endpoint,
flag, or committed smoke script.

### Interpretation limits

Twenty measured requests per case make p95 exploratory, not a stable production
tail estimate. The machine was not certified free of other activity; fixed case
order and thermal state can affect comparisons. Process RSS is a lifetime
high-water mark, not per-case allocation or unified/device memory. Token timing
and metrics output add overhead. No MPS comparison, queue, concurrency change,
or performance optimization is justified by this single baseline run.

### Completion and next lesson

The seven requested Phase 3 implementation steps are complete and exercised.
README has repeatable commands and metric definitions; `plan.md` reflects the
verified technical checklist. All benchmark/harness processes started for this
work exited. No production source, package versions, generated changelogs,
`.ua` artifacts, staging, or commits were changed.

Next lesson: use the table to explain why client first text differs from model
TTFT, why longer inputs mainly affect prefill, why mean TPOT differs from pooled
inter-token percentiles, and why matrix-kernel dominance does not by itself
prove memory-bandwidth saturation. The learner's explanations remain to be
verified; Stage 4 implementation is not implicitly authorized.

## Phase 4: bounded concurrency and cancellation

### Authorization and scope

The learner explicitly requested implementation after reviewing the controlled
concurrency plan. This supersedes the earlier instruction not to start Stage 4
implicitly. It does not authorize Stage 5 batching or a different serving stack.

The CPU float32 Qwen runtime, model revision, request validation, context budget,
generation settings, and one-model-worker baseline remain unchanged. There is no
batching, parallel model execution, queue TTL, global generation deadline, or
attempt to kill Python/native threads.

### Implemented policy

- One running job plus two event-loop-owned FIFO waiting slots. Only the active
  job is submitted to the executor; the executor is not a second waiting queue.
- Full capacity returns HTTP 429 `server_busy`. `/health` stays 200 while HTTP is
  reachable. `/ready` reports running/queued/capacity fields: idle or busy with
  space is 200, full is 503, and unavailable/shutting down is 503.
- Scheduling transitions are `queued -> running -> completed/failed/canceled`
  or `queued -> canceled`. Delivery outcomes are separate.
- A queued disconnect removes the job without tokenization/model execution.
  A running disconnect signals cooperative cancellation; only actual worker
  completion releases ownership and promotes the next waiting job.
- Cancellation is checked around preparation and through a Transformers
  stopping criterion. It is not a successful truncated completion and produces
  no invented completion usage. The first reason/timestamp wins.
- Text buffers hold at most eight items, including the sentinel. Empty text
  increments are skipped. A full buffer has a one-second stall limit with
  cancellation checks at most 50 ms apart. ASGI sends have a five-second limit.
  A writable stream can emit a safe error and `[DONE]`; a blocked/disconnected
  transport has no such guarantee.
- Normal Uvicorn shutdown closes the listener and drains HTTP before lifespan
  teardown. Teardown cancels any remaining waiting jobs, signals the active job,
  and retains runtime ownership until the worker really exits. A stuck native
  call can still prevent shutdown.
- Settings and stream buffers are per request. `seed` remains unsupported;
  sampling uses serialized process RNG state without request-specific global
  resets.
- `request_started` records queue wait. Every admitted job has a
  `scheduling_finished` event, including queued cancellation and submission
  failure. Running cancellation delay is signal-to-worker-exit, not socket-close
  time. Never-started jobs have null queue wait/cancellation delay.

### Regression and real-model controls

`.venv/bin/python -m unittest discover -s tests -q` passed all 106 tests. Coverage includes
actual tiny-HF stopping, FIFO promotion after cancellation/failure, abandoned
future observation, bounded queues, transport deadlines, iterator cleanup,
shutdown, independent response state, and benchmark accounting.

An initial benchmark regression assumed cancellation checks followed streamer
callbacks. Installed Transformers 5.18.0 checks stopping before forwarding that
step's token. The incidental exact callback-count/TPOT assumptions were removed;
the test still requires canceled classification without fabricated completion
usage. Production behavior was not changed to fit that assumption.

A separate localhost Uvicorn server used the real cached Qwen revision
`7ae557604adf67be50417f59c2c2f167def9a775`. The temporary control wrapped real
forward calls with an event gate; it did not replace model output. Saved evidence:
`benchmark-results/phase4/smoke.json`.

| Control | Observed result |
|---|---|
| One held active job plus two waiting | Fourth request 429; health 200; readiness 503 `full`, running 1, queued 2 |
| Queued TCP disconnect | Removed before worker start; readiness became 200 `busy`, queued 1; replacement admitted |
| Running TCP disconnect during held real forward | Ownership retained; neither waiting job started before release; cancellation delay 111.57 ms, including the deliberate hold |
| FIFO recovery and independent output caps | Remaining jobs completed in order with four and six generated tokens; queue waits 278.15 and 550.30 ms |
| Streaming TCP close after first real content | Canceled producer; signal-to-exit delay 84.31 ms in this observation |
| Controlled blocked ASGI send | Buffer reached eight items; producer canceled as `slow_consumer`; delivery ended after 5006.17 ms without a terminal body |
| Normal Uvicorn graceful shutdown | Listener refused new connections; active and queued requests both finished HTTP 200 before runtime release |
| Direct lifespan teardown with work present | Queued and active requests returned 503 `not_ready`; queued job never started; runtime retained through the held worker; active delay 112.50 ms including the hold |

Peak simultaneous real model forwards was one. The slow-send case was controlled
ASGI pressure with real Qwen, not proof of saturating the OS TCP send buffer.
Its producer signal-to-exit delay was 0.046 ms because the streamer itself raised
on the full-buffer deadline. The one-second stall wait precedes that signal and
must not be confused with cancellation delay. These individual control timings
are observations, not service-level guarantees.

### Completion-boundary race correction

A real-Qwen HTTP probe exposed an additional thread interleaving: the worker
exited after `cancel_inference` checked its state but before the cancellation
token recorded its signal. Before the correction, successful work was relabeled
`canceled` and emitted a negative cancellation delay of -0.036 ms.

Worker-exit ordering is now authoritative. A signal timestamped after worker
exit is delivery-only: it cannot relabel the producer or create producer
cancellation metrics. The repeated real-HTTP probe preserved `completed`, four
actual generated tokens, and null cancellation delay, with no cancellation
reason on terminal scheduling/producer records. A deterministic regression
keeps this interleaving covered; all 106 tests pass.

Before/after evidence is saved as `exit-race-before.json` and
`exit-race-after.json` under `benchmark-results/phase4/`. The temporary model
processes exited and the control script was removed.

### Fixed-workload concurrency comparison

The existing `benchmark-results/phase3/workload.json` was reused unchanged:
59 or 990 prompt tokens, crossed with 32 or 128 output tokens, greedy decoding,
and the same pinned Qwen/tokenizer revision and template hash. Each concurrency
level used all four cases, two sequential warm-ups per case, and exactly 20
measured requests per case. Requests arrived in bounded waves, without retries.

One isolated instrumented server on localhost port 54415 served C1, C2, then C4
sequentially. Recorded host/runtime: Apple M1 Pro, eight logical CPUs, CPU
float32, PyTorch intra-op 6/inter-op 8, Python 3.12.13, PyTorch 2.14.1,
Transformers 5.18.0, tokenizers 0.23.2, FastAPI 0.143.3, Uvicorn 0.54.0, and
HTTPX 0.28.1. Cached loading took 1637.08 ms; lifespan-to-instrumentation startup
took 1687.28 ms, excluding imports. Neither is included in request measurements.

All 220 successful measured requests reached their requested cap with
`finish_reason: length`. C1 and C2 each completed 80/80 requests. C4 completed
60/80 and rejected 20 with HTTP 429, exactly five rejections per case. There
were no generation/client errors or warm-up failures. The C4 client exited 1
because it records rejections as non-success, not because evidence was lost.

Times below are milliseconds. Latency percentiles describe successful requests,
not the much faster rejected requests.

| Input/output tokens | Client C | Success/requests | Queue wait p50/p95 | E2E p50/p95 | Useful output tok/s |
|---|---:|---:|---:|---:|---:|
| 59/32 | 1 | 20/20 | 0.15 / 0.22 | 2459.20 / 2522.83 | 12.98 |
| 59/32 | 2 | 20/20 | 1214.84 / 2485.03 | 3731.69 / 5046.47 | 12.92 |
| 59/32 | 4 | 15/20 | 2438.14 / 4928.92 | 4880.99 / 7400.76 | 13.05 |
| 59/128 | 1 | 20/20 | 0.15 / 0.27 | 9805.63 / 10335.64 | 12.99 |
| 59/128 | 2 | 20/20 | 4808.15 / 10299.49 | 15074.85 / 20108.09 | 12.94 |
| 59/128 | 4 | 15/20 | 9739.96 / 19582.04 | 19460.92 / 29675.59 | 13.01 |
| 990/32 | 1 | 20/20 | 0.16 / 0.23 | 3366.76 / 3755.43 | 9.43 |
| 990/32 | 2 | 20/20 | 1557.58 / 3286.59 | 4808.26 / 6555.05 | 10.04 |
| 990/32 | 4 | 15/20 | 3138.68 / 6444.43 | 6293.28 / 9817.34 | 10.03 |
| 990/128 | 1 | 20/20 | 0.16 / 0.35 | 11090.29 / 11555.24 | 11.47 |
| 990/128 | 2 | 20/20 | 5369.11 / 11421.02 | 16917.69 / 22341.25 | 11.54 |
| 990/128 | 4 | 15/20 | 10914.14 / 22438.89 | 22060.04 / 33206.76 | 11.63 |

For 59/32, model TTFT p50 stayed at 120.67 / 123.01 / 122.79 ms for C1/C2/C4,
while client first-text p50 rose to 198.26 / 1418.81 / 2638.45 ms. Model TPOT p50
was 75.18 / 74.85 / 74.69 ms. This separates queueing from model execution:
more connected clients did not create parallel model calls or a proportional
throughput gain. The 990/32 throughput variation is not enough to attribute a
speedup to concurrency in this single fixed-order run.

Recorded peak RSS ranged from 2137.27 to 2139.83 MiB. It is process-lifetime
high-water RSS, not per-request memory or current device allocation.

Raw artifacts: `http-c1.json`, `http-c2.json`, `http-c4.json`, `server.jsonl`,
and `report-c1.json`, `report-c2.json`, `report-c4.json`, all under ignored
`benchmark-results/phase4/`. Reports joined every successful request to model
and scheduling records without missing-metric warnings. README documents the
same commands and flags for a fresh output directory.

The uncanceled comparison preceded the final late-signal race correction.
That correction changes only cancellation-outcome classification/metrics;
these uncanceled cases did not exercise it. Post-correction cancellation and
client controls use a fresh server and separate metrics file.

These are bounded-wave observations, not constant-arrival-rate or mixed-length
queue tests. C1/C2 have 20 successful samples per case, C4 only 15; p95 remains
exploratory. Fixed run order, thermal state, background activity, and opt-in
instrumentation can affect the numbers. There is no hard end-to-end deadline:
cooperative cancellation still cannot interrupt a stuck native call.

### Post-correction cancellation and client controls

A fresh instrumented server used `control-server.jsonl`, separate from the
uncanceled comparison. For `short-input-long-output`, two normal warm-ups were
followed by 20 measured C1 streams closed after their first real text event:

- Client and server both recorded 20 cancellations, with no errors, rejections,
  or missing terminal cancellation measurements.
- Signal-to-worker-exit delay: p50 **74.51 ms**, p95 **80.63 ms**, minimum
  **72.67 ms**, maximum **143.22 ms**.
- Client and model cancellation records had no completion-token count; client
  rows had no successful `stop`/`length` finish reason. Partial output was not
  promoted into completed-request throughput.
- Queue wait was p50 **72.76 ms**, p95 **79.01 ms**, despite client C1. Closing
  a response lets that client issue its next request before the previous native
  worker exits. The next job correctly waits for cancellation completion.
- These delays begin when the server records its signal. Disconnect-detection
  time is additional; this is not a client-close-to-idle service guarantee.

Artifacts: `canceled.json`, `report-canceled.json`, and `control-server.jsonl`
under `benchmark-results/phase4/`.

Real HTTP isolation controls used two different prompts and output caps. Their
concurrent responses exactly matched their own serial baselines: `ORANGE` and
`violets`, each with three generated tokens. The second concurrent request waited
261.95 ms before starting. This verifies response isolation, not correctness of
the model's instruction following.

A further concurrent greedy/sampled pair reached the actual model with separate
configurations: greedy `do_sample:false`, cap 4; sampled `do_sample:true`,
temperature 0.7, top-p 0.9, top-k 0, cap 8. An unsupported `seed` request returned
422 `invalid_request`. Details and request IDs are in `isolation.json`; effective
generation settings are in the correlated control-server records.

The actual benchmark CLI was interrupted with SIGINT while one request ran and
another waited. It saved both rows as canceled, set `interrupted:true`, and exited
130. The server returned idle. `interrupted.json` and `report-interrupted.json`
preserve this partial run rather than pretending all 20 planned requests ran.

One interrupted client had no response request ID because it disconnected while
queued, before headers. Its server scheduling record still proves cancellation
before worker start. The report's generic unavailable-cancellation warning is
therefore expected: the missing client/server join is not evidence of a worker
still running, and no zero-valued timing was invented.

### Completion and next lesson

Phase 4 implementation and technical verification are complete. README covers
the current API contract and repeatable benchmark commands; the roadmap records
technical completion without checking off independent understanding.
All server/control processes started for this work exited, and temporary smoke
scripts were removed. No dependencies, generated changelogs, `.ua` files,
staging, or commits were changed.

Before batching, the learner should explain:

1. Why client first text grows with queue wait while model TTFT stays similar.
2. Why a busy model can still be ready to accept a waiting request.
3. Why a disconnected HTTP coroutine cannot release native worker ownership.
4. Why C1 cancellation traffic can still have queue wait.
5. Why C4 success percentiles must be read alongside its 25% rejection rate.
6. Why normal Uvicorn shutdown drains HTTP, while direct lifespan teardown
   cancels work left behind.

These explanations remain unverified. Stage 5 implementation is not implicitly
authorized by completion of this phase.
