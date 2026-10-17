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

The next milestone discussed is a minimal single-worker HTTP server using
Transformers `generate()`, retaining the manual scripts as learning references.
Manual top-k/top-p lessons were explicitly skipped, not completed. No server
has been implemented or authorized by this documentation update.
The stable roadmap is [plan.md](../plan.md); its checkboxes are not a live
completion ledger. Use this handoff and the actual code for the current state.

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

Next task: review Checkpoint 4's existing generation-parameter mapping with the
learner, then connect the request contract to the later application lifecycle,
admission path, and error responses. The Python validation portion is exercised,
not an HTTP API. Existing educational scripts and inference code are unchanged
by this checkpoint.

Still not established: a formal temperature study, manual top-k/top-p filtering
(explicitly skipped), repeated statistical benchmarks, total/peak process-memory
profiling, actual concurrency/batching experiments, precision/quantization
comparisons, or any HTTP server. Basic Stage 1 milestones have been exercised
with that sampling scope decision; do not claim the entire roadmap is complete.

