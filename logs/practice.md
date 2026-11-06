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
bounded scheduling/cancellation are implemented. There are 131 passing tests.
Real-Qwen controls verify FIFO admission, queued removal, running disconnects,
bounded streaming pressure, request-setting isolation, and both shutdown paths.
See [the Phase 4 entry](#phase-4-bounded-concurrency-and-cancellation) for current
policy, measured concurrency evidence, and limitations.
Stage 5 now has completed CPU and T4 static-batching measurements.
Stage 6 now has completed five-case CPU and T4 dynamic KV-cache scaling studies.
All T4 cases matched predicted logical bytes and their same-device HF references;
saved-file comparison also found exact CPU/GPU cache-byte and token-ID agreement.
A CPU dynamic/static allocation comparison also passed with identical generated
IDs and measured capacity differences. Next: interpret unused static capacity,
then introduce paged allocation conceptually, without implementing an allocator.
Manual top-k/top-p lessons were explicitly skipped, not completed. Use the
stable roadmap [plan.md](../plan.md), this evidence log, and actual code together.
The latest checkpoint is the
[CPU dynamic/static allocation result](#cpu-dynamicstatic-cache-allocation-2026-11-05).
The original CPU study and earlier T4 batching results remain preserved.

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
- Standing request (2026-10-27): record future experiment numbers and results
  so progress can be compared at the end of the project. Preserve available
  per-run data under fresh local `benchmark-results/` paths and summarize
  observed evidence here with configuration, timing boundaries, sample counts,
  failures, and limitations. Treat learner-supplied output as evidence; do not
  rerun merely to confirm it or invent missing measurements. Separate functional
  success from performance evidence. Compare matching workloads and boundaries,
  reporting improvements, regressions, or inconclusive results honestly.
  This authorizes recording evidence, not unapproved code changes or cloud runs.

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
- `src/cached_generation.py`: import-safe `generate_cached` reuses the manual
  prefill/decode loop for left-padded Qwen batches with independent EOS stopping.
  Returns generated IDs, finish reasons, and predicted/measured cache snapshots.
  The standalone Hello demonstration retains CPU/MPS selection, float32,
  discarded prefill warm-up, synchronized timings, and Hugging Face ID parity.
  Dense finished rows stay in the cache until the whole batch finishes.
  On CUDA the helper also owns peak-memory resets after warm-up release and
  captures synchronized allocator peaks before any reference generation.
  Optional `cache_capacity` selects native static allocation. Cache snapshots
  distinguish processed slots, the library's raw length query, and tensor capacity.
  Explicit write positions avoid inferring offsets from nonzero cache contents.
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
- `src/batched_generation.py`: standalone greedy static-batch generation and an
  offline CPU comparison against individual calls, with preserved JSON evidence.
- `benchmarks/batching.py`: fixed four-prompt batch-size 1/2/4 timing, actual-token
  throughput, output comparisons, and CUDA allocator peaks when running on CUDA.
- `modal_batching.py`: separate bounded T4 entrypoint; the first run is recorded below.
- `benchmarks/cache_scaling.py`: offline pinned CPU/CUDA float32 study of three
  prompt lengths and batch sizes 1/2/4, with HF comparisons, logical KV bytes,
  CUDA allocator peaks, and fresh JSON evidence. CPU remains the default.
  `--compare-static` opts into a CPU-only, batch-1 comparison of dynamic cache
  and static capacities 160/256 on the same prompt. Default dynamic studies remain.
- `modal_cache_scaling.py`: bounded T4 entrypoint reusing the pinned image;
  the first real T4 cache-scaling run is now recorded below.

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

## GPU exploration: Modal and warm inference

### Authorization and position

On 2026-10-27, the learner approved a bounded warm GPU experiment and requested
ongoing recording of numbers and results. This is an exploration after Phase 4,
not Phase 5 batching or API deployment. The local CPU service, its generation
implementation, and CPU-only token instrumentation remain unchanged. Cloud
execution remains a separate learner action; no always-warm GPU is configured.

### User-reported CUDA functionality

The learner ran the initial tensor check on a Tesla T4 with Torch 2.6.0+cu124
and CUDA 12.4. Its CUDA tensor returned squares `[0, 1, 4, 9, 16, 25, 36, 49]`;
reported device capacity was 14.57080078125 GiB, not memory consumption.

The first Qwen invocation built the image and downloaded seven artifact files.
Model placement and generation succeeded, but result reporting failed because
`importlib.metadata.version("modal")` could not find distribution metadata for
Modal's injected runtime. Removing that lookup fixed the reporting assumption,
without changing inference or installing another SDK in the container.

The learner's subsequent successful run:
https://modal.com/apps/lenando0307/main/ap-bfypt22VLnBnj9tpKOQjkh

| Field | Observed value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Revision | `7ae557604adf67be50417f59c2c2f167def9a775` |
| Template SHA-256 | `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` |
| Device / dtype | Tesla T4, `cuda:0`, `torch.float32` |
| Python / CUDA | 3.12.10 / 12.4 |
| Torch / Transformers | 2.6.0+cu124 / 4.57.6 |
| NumPy / Hub / tokenizers / safetensors | 2.2.6 / 0.36.0 / 0.22.1 / 0.6.2 |
| Generation settings | Output cap 16, temperature 0, top-p 1 |
| Completion | `4` |
| Prompt / completion tokens | 43 / 2 |
| Finish reason | `stop` |

Preserved as ignored local evidence:
`benchmark-results/modal/qwen-check-user-reported.json`.
This is learner-supplied output, not an assistant rerun. No model latency or
speedup was measured, and the shell prompt's elapsed time is not an inference
measurement. This establishes functionality, not independent understanding.

### Implemented warm experiment

- `modal_inference.py` loads one pinned CPU float32 runtime, moves its weights
  to CUDA once, and invokes the helper loop on the same runtime.
- The original `benchmarks/gpu.py` (now `benchmarks/warm.py`) reuses
  `benchmarks.model.validate_workload` to check the recorded revisions, template,
  actual tokenization, and context budget. It selects only
  `short-input-long-output`: 59 prompt tokens and a 128-token cap.
- Two warm-ups precede five sequential measured calls. Each call has separate
  generation state and KV cache. Natural EOS stopping is unchanged.
- `warm_inference_ms` brackets the complete `src.inference.generate()` call
  with CUDA synchronization before the starting clock read and before the
  ending clock read. It includes formatting, tokenization, transfers, prefill,
  decoding, and text decoding. It excludes loading, warm-ups, network invocation,
  and gaps between calls. No per-token instrumentation is added.
- The JSON retains environment metadata, the full workload, phase/iteration,
  status, actual counts, and finish reason for each returned sample. Warm-up
  duration is null. Summary statistics exclude warm-ups and failed calls.
- Throughput is total successful measured output tokens divided by their summed
  warm inference seconds. It is not decode-only throughput or whole-run
  throughput. Five observations do not support a stable p95 or production SLO.
- An ordinary sample failure aborts subsequent calls, returns earlier rows, and
  records only the exception type with null failed completion usage and timing.
  The local CLI saves incomplete results and exits 1; handled Ctrl-C exits 130.
- A transport, setup, or process failure before results return leaves remote
  history unknown: `remote_results_received=false`, no available samples, and
  `summary=null`. No zero-valued remote execution counts are invented. Forced
  local termination or build failure before entrypoint execution cannot be
  saved by the entrypoint; preserve those CLI logs separately.
- Existing paths are refused, and the shared `write_json` helper creates files
  exclusively. Benchmark modules are mounted after the cached download layer;
  only the workload JSON is sent, not the result directory.

Prediction recorded before any remote timing run: keeping one loaded model
excludes repeated model initialization from the measured calls. It does not
establish a numerical GPU speedup. The remote software stack differs from the
historical CPU stack, and GPU warm function duration cannot be compared
directly with CPU HTTP E2E duration.

### Local verification and numbers

The full existing environment suite passed **110 tests**. Four new deterministic
tests cover workload mismatches, leading/trailing synchronization timing,
warm-up and generation failures, and token-weighted throughput with warm-ups
and errors excluded. The asynchronous timing test models pending GPU work;
it is not evidence of real CUDA timing behavior.

A real pinned-Qwen CPU smoke exercised the actual two-warm-up/five-measurement
loop and selected workload. Only CUDA synchronization was replaced with a no-op
for this local verification; the Modal worker itself still rejects a CPU host.
This is **CPU validation, not the GPU benchmark or a speedup comparison**.

Environment: Apple M1 Pro, eight logical CPUs, Python 3.12.13, Torch 2.14.1,
Transformers 5.18.0, float32, intra-op threads 6 and inter-op threads 8.
Both warm-ups succeeded; all five measured calls used 59 prompt tokens and
generated 128 tokens with `finish_reason: length`. No generation failures.

| Measured iteration | CPU validation duration (ms) |
|---|---:|
| 0 | 9422.451042 |
| 1 | 9267.304792 |
| 2 | 9222.209834 |
| 3 | 9695.879542 |
| 4 | 10477.113125 |

Median: **9422.451042 ms**. Sum of measured durations: **48084.958335 ms**.
Actual measured output: **640 tokens**, yielding **13.309775 tok/s** at the
defined successful-call boundary. These numbers are preserved because the
learner requested recording all observed results, not because they establish
an optimization.

Raw artifact: `benchmark-results/modal/local-cpu-validation-b262b6e5.json`.
Its kind and verification limit explicitly identify CPU execution and the
replaced synchronization. Local artifacts are ignored and may be absent in
a fresh clone.

The real local entrypoint and writer also passed controlled transport-boundary
checks: successful JSON persistence, byte-for-byte preservation on overwrite
refusal, partial generation failure preservation, transport failure, and
interruption exit 130. Network calls were replaced in those checks; they are
not remote lifecycle evidence. Temporary persistence-check files were removed.
Modal SDK 1.6.2 CLI help confirms required `--output` and optional `--workload`.

### First warm GPU invocation (completed)

From the repository root, only when ready to consume cloud credits:

```bash
~/.venvs/modal/bin/modal run modal_inference.py \
  --workload benchmark-results/phase3/workload.json \
  --output benchmark-results/modal/warm-t4-001.json
```

Choose another fresh output filename if it already exists. The GPU allocation
is billed during loading, warm-ups, measured work, and any remaining container
lifetime. One allowed container, no configured input retries or minimum warm
capacity, and timeouts are not hard spending caps.

At implementation handoff, remote timing was still pending. No cloud computation,
deployment, dependency changes, generated-file edits, staging, or commits were
performed by the assistant during that implementation. The learner subsequently
ran the command successfully; its evidence follows.

### First warm T4 measurements: 2026-10-27

Source: learner-provided terminal output and the saved local artifact
`benchmark-results/modal/warm-t4-001.json`. The assistant did not rerun it.
Recorded start: `2026-10-27T23:53:18.465258+00:00`.
Modal run: https://modal.com/apps/lenando0307/main/ap-iktRlOMFzQ62SxDC3YJt4t
The CLI reported the result saved and the ephemeral app completed.

Configuration: the same pinned model/tokenizer revision and template hash as
above; Tesla T4, CUDA float32, Python 3.12.10, Torch 2.6.0+cu124,
Transformers 4.57.6, NumPy 2.2.6, Hub 0.36.0, tokenizers 0.22.1, safetensors
0.6.2. Batch size and concurrency were one. One model load, two warm-ups,
five sequential measured calls; greedy temperature 0/top-p 1, natural EOS
stopping, 128-token output cap, and 2048-token context limit.

Both warm-ups and all five measured calls succeeded. Each used 59 prompt
tokens and produced 128 completion tokens with `finish_reason: length`.
There were zero warm-up errors and zero measured errors. The full workload
manifest contains four cases, but only `short-input-long-output` ran.

| Measured iteration | Warm T4 inference duration (ms) |
|---|---:|
| 0 | 3126.420059 |
| 1 | 3924.504596 |
| 2 | 3890.214294 |
| 3 | 3397.125489 |
| 4 | 3025.067633 |

- Median: **3397.125489 ms**.
- Observed range: **3025.067633 to 3924.504596 ms**.
- Sum of measured inference durations: **17363.332071 ms**.
- Measured output: **640 tokens**.
- Aggregate successful measured throughput: **36.859285 output tokens/s**,
  calculated as `640 / 17.363332071`.

The measured boundary includes formatting, tokenization, transfers, prefill,
decoding, and text decoding, with CUDA synchronization at the outer boundaries.
It excludes loading, warm-ups, network invocation, and gaps between calls.
The sum is neither total command duration nor billable GPU allocation duration.
The workload's `baseline.device: cpu` describes its original manifest; the
actual execution device is `runtime.device: cuda:0`.

#### Exploratory cross-environment comparison

The earlier CPU validation used the same fixed workload, revision, float32
precision, two/five sample structure, and analogous whole-function timing
boundary. Both runs generated 128 tokens on every measured call, so shorter
output does not explain the observed difference. The CPU synchronization
replacement was a no-op because those operations executed on CPU.

| Metric | Local M1 Pro CPU validation | Modal T4 |
|---|---:|---:|
| Measured successes | 5/5 | 5/5 |
| Median warm inference (ms) | 9422.451042 | 3397.125489 |
| Aggregate output tokens/s | 13.309775 | 36.859285 |
| Sum of measured durations (ms) | 48084.958335 | 17363.332071 |
| Torch / Transformers | 2.14.1 / 5.18.0 | 2.6.0+cu124 / 4.57.6 |

Arithmetic on these observed summaries:

- CPU median / T4 median: **2.773654x**, or **63.946478% lower median duration**.
- T4 aggregate throughput / CPU aggregate throughput: **2.769339x**.

These are observed cross-environment ratios, not a hardware-isolated GPU
speedup claim. Python/package versions, host CPUs, operating systems, kernels,
and execution conditions differ; the CPU run was originally a functional
validation smoke, not a controlled matched-stack benchmark. There are only
five measured samples in each single run. Timing variation is observed, but
its cause is not established by these records. No stable p95, TTFT, TPOT,
cold-start latency, quality improvement, or service-level speedup is established.
Do not compare against Phase 4 HTTP E2E figures or infer GPU billing from the
summed warm durations.

A defensible statement is: on the fixed 59/128 workload, this T4 environment
observed a 3.40-second median warm inference duration and 36.86 output tokens/s,
versus 9.42 seconds and 13.31 tokens/s in the earlier local CPU validation,
with software and platform differences explicitly disclosed.

This motivated the separately approved matched-library CPU control below.
It removes the six core-package release mismatches, but does not by itself
isolate hardware. The supplied GPU result needs no confirmation rerun.

### Matched-library CPU control: 2026-10-28

After committing the bounded GPU experiment as `828a39e`, the learner approved
the proposed runner changes, separate local environment, and bounded CPU run.
The project `.venv`, requirements, and Modal SDK environment were not changed.
No Modal job or image build was launched; the comparison reuses the existing
`benchmark-results/modal/warm-t4-001.json`.

#### Implementation and reproducible command

- Renamed `benchmarks/gpu.py` to `benchmarks/warm.py` and migrated the Modal
  imports/mount and tests. The shared loop uses synchronous CPU execution or
  device-specific CUDA synchronization around the same `src.inference.generate()`
  boundary. Unsupported accelerators are rejected. The Modal CUDA/float32
  guards remain mandatory.
- Shared runtime metadata records the six core packages, revision/template,
  generation settings, device/dtype, host, and Torch thread counts.
- The CPU CLI requires `HF_HUB_OFFLINE=1`, loads the pinned model once in CPU
  float32, sets six intra-op threads, validates the fixed case, then performs
  two warm-ups and five sequential measurements with natural EOS enabled.
  It uses real CPU execution, not a CUDA synchronization mock.
- Existing evidence cannot be overwritten. Ordinary sample failures retain
  earlier rows; failure before the loop returns leaves `samples_available=false`
  and a null summary, not fabricated measurements. Handled failures exit 1
  and interruption exits 130; forced termination cannot save a record.

Created `/Users/nando/.venvs/inference-cpu-control` with the project's Python
3.12.13, then installed exactly:

```bash
~/.venvs/inference-cpu-control/bin/python -m pip install \
  torch==2.6.0 transformers==4.57.6 numpy==2.2.6 \
  huggingface-hub==0.36.0 tokenizers==0.22.1 safetensors==0.6.2
```

The local Torch wheel is the native macOS CPU/MPS-capable build, not
`2.6.0+cu124`. Its release version matches; the runner explicitly uses CPU.
All six installed release versions were checked against the saved GPU metadata,
and `pip check` reported no broken requirements.

Executed from the repository root after regression checks had finished:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.warm \
  --workload benchmark-results/phase3/workload.json \
  --output benchmark-results/modal/cpu-matched-001.json
```

Saved local evidence: `benchmark-results/modal/cpu-matched-001.json`, recorded
at `2026-10-28T00:30:38.754763+00:00`. The command completed successfully.
Raw benchmark files remain ignored local artifacts, not committed fixtures.

#### Observed CPU environment and samples

- Host: Apple M1 Pro, arm64, eight logical CPUs; recorded platform
  `macOS-26.6.2-arm64-arm-64bit`.
- Python 3.12.13; Torch 2.6.0, Transformers 4.57.6, NumPy 2.2.6,
  Hugging Face Hub 0.36.0, tokenizers 0.22.1, safetensors 0.6.2.
- Device `cpu`, dtype `torch.float32`, no CUDA runtime, six intra-op threads,
  eight inter-op threads.
- Model `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`; chat-template SHA-256
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Same complete workload manifest and selected 59/128 case as the saved T4 run;
  float32, temperature 0, top-p 1, context limit 2048, batch/concurrency one.
- Both warm-ups and all five measured calls succeeded. Every call reported
  59 prompt tokens, 128 completion tokens, and `length`. Warm-up timing is null.
  There were zero warm-up or measured errors.

| Measured iteration | Warm inference (ms) | Prompt tokens | Completion tokens | Finish |
|---|---:|---:|---:|---|
| 0 | 11604.624167 | 59 | 128 | length |
| 1 | 12106.442000 | 59 | 128 | length |
| 2 | 11430.556500 | 59 | 128 | length |
| 3 | 11536.932000 | 59 | 128 | length |
| 4 | 11690.637875 | 59 | 128 | length |

Median warm inference: **11604.624167 ms**. The 640 measured completion tokens
over **58369.192542 ms** of summed successful inference time yielded
**10.964688 output tokens/s**. This includes the entire generation function,
not just decoding; it excludes loading, warm-ups, gaps between calls, and
network invocation.

#### Comparison without rerunning the GPU

| Metric | Matched-library M1 Pro CPU | Existing Modal T4 |
|---|---:|---:|
| Measured successes | 5/5 | 5/5 |
| Median warm inference (ms) | 11604.624167 | 3397.125489 |
| Aggregate output tokens/s | 10.964688 | 36.859285 |
| Summed measured duration (ms) | 58369.192542 | 17363.332071 |
| Measured completion tokens | 640 | 640 |
| Torch | 2.6.0 | 2.6.0+cu124 |
| Transformers | 4.57.6 | 4.57.6 |
| Python | 3.12.13 | 3.12.10 |

Arithmetic from the saved, unrounded values:

- CPU median / T4 median: **3.416013x**.
- T4 median duration reduction relative to CPU: **70.726105%**.
- T4 aggregate throughput / CPU aggregate throughput: **3.361635x**.
- Relative to the earlier CPU validation, this CPU median is **23.159294%
  higher**, and aggregate throughput is **17.619285% lower**. Preserve that
  regression in the observations; do not assume matching versions improves speed.

The six core release versions now match, and the workload, revision, template,
precision, generation settings, sample structure, actual counts, and finish
reasons were checked for equality. The CPU and CUDA branches bracket the same
whole generation function. Both runs reached the cap, so shorter completion
length does not explain the duration difference. Matching counts does not
establish identical generated token IDs or text; those were not retained.

This remains a cross-platform, single-run comparison, not a hardware-only
causal measurement. Torch builds, Python patch versions, CPU/GPU kernels,
host hardware and operating systems differ; transitive dependencies and
execution conditions were not fully controlled. The prior CPU run also used
different core versions and was collected at a different time, so these records
do not establish why its latency was lower. Five samples per run do not establish
a stable p95, HTTP latency, TTFT, TPOT, cold-start time, quality change, cost,
or production SLO. No extra run is needed merely to confirm these observations.

#### Verification

- `.venv/bin/python -m unittest discover -s tests -v`: **113 passed**.
- `~/.venvs/inference-cpu-control/bin/python -m unittest discover -s tests -p test_benchmark_warm.py -v`:
  **7 passed** under the matched core versions.
- New checks exercise real tiny-model CPU generation with CUDA access forbidden,
  reject unsupported accelerators, and verify safe failure evidence and
  overwrite refusal. Existing CUDA clock-boundary, failure, workload, and
  token-weighted throughput checks remain passing.
- The actual offline Qwen command above completed two warm-ups plus five
  measurements and saved its result; this is the runtime smoke, not just unit tests.
- The Modal function was invoked only via `.local()` on the actual CPU runtime:
  it rejected execution with `This experiment requires CUDA.` before loading.
  `modal run modal_inference.py --help` also succeeded without a cloud invocation.
- A second CLI invocation against the existing output path exited 2 before
  model loading; its SHA-256 stayed unchanged. This exercised overwrite refusal,
  not another benchmark run.
- GPU hardware execution of the refactored runner was not repeated. The GPU
  performance reference remains the learner's previously completed run.

## Stage 5: static batching CPU correctness

### Approved scope and engineering decision

On 2026-11-02, the learner approved implementing and running the CPU-first
batching correctness experiment. Add a separate learning script and focused
checks; do not modify the single-request API, its scheduler, the existing warm
benchmark, or the Modal entrypoint. No cloud run was authorized or launched.
The existing project and matched-library environments were reused without
dependency changes.

`plan.md` remains the stable roadmap, not a live completion ledger. This
checkpoint starts Stage 5; it does not complete its throughput/memory study.

### Implemented behavior

- Added `src/batched_generation.py::generate_batch(runtime, conversations,
  max_new_tokens=...)`. It reuses `Runtime` and `GenerationResult`, applies the
  existing tokenizer's chat template independently, and uses per-call left
  padding plus attention masks without changing shared tokenizer configuration.
- Input order is preserved. Prompt counts are mask sums; completion extraction
  starts after the common padded input width, not each unpadded prompt length.
- One Transformers generation call handles the batch with the same explicit
  greedy settings as the existing single-request function. Model weights are
  reused; each call has fresh generation/cache state.
- Empty conversations and invalid output caps are rejected. The maximum padded
  input width plus output budget must fit the runtime context limit; no input
  is silently truncated.
- Each completion ends at its first generated configured EOS, inclusive for
  usage accounting. Post-completion padding is discarded even when pad and EOS
  share an ID. Non-EOS completions at the cap report `length`.
- Added `main()`: load the pinned CPU float32 runtime once, run A then B through
  unchanged `src.inference.generate()`, then run `[A, B]` through the new helper.
  Print and save actual IDs, text, counts, finish reasons, and equality checks.
- Reused the shared metadata and exclusive JSON writer. A completed comparison
  can report unequal outputs; `all_token_ids_equal` is separate from execution
  status. Failures preserve previously returned reference results and only
  the exception type, not private exception text. Ctrl-C is marked interrupted.

No performance timers, batching wait window, streaming batch support, automatic
API request collection, continuous batching, or new inference engine were added.

### Real CPU execution

Command executed from the repository root:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m src.batched_generation \
  --max-new-tokens 32 \
  --output benchmark-results/batching/cpu-correctness-001.json
```

Recorded at `2026-11-02T18:56:11.720358+00:00`; status `completed`.
Evidence is an ignored local artifact at the path above, not a committed fixture.

Configuration:

- Qwen/Qwen2.5-0.5B-Instruct, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Chat-template SHA-256:
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Apple M1 Pro CPU, arm64, eight logical CPUs, float32.
- Python 3.12.13; Torch 2.6.0, Transformers 4.57.6, NumPy 2.2.6,
  Hugging Face Hub 0.36.0, tokenizers 0.22.1, safetensors 0.6.2.
- Six intra-op and eight inter-op threads; context limit 2048; greedy decoding
  with temperature 0 and top-p 1; maximum 32 new tokens per sequence.
- Two individual reference calls and one static batch-size-2 call, using the
  same model instance and conversations. No warm-up or performance measurement.

Inputs:

1. `Reply with exactly one word: Hello`
2. `Explain how an inference server processes a request. Use a numbered list covering tokenization, prefill, decoding, the KV cache, and stopping.`

| Input | Actual prompt tokens | Individual completion tokens | Batched completion tokens | Finish, both paths | Exact generated-ID match |
|---|---:|---:|---:|---|---|
| A: Hello | 36 | 10 | 10 | stop | yes |
| B: inference explanation | 59 | 32 | 32 | length | yes |

Each execution mode produced 42 actual completion tokens. Prompt counts and
finish reasons also matched for both inputs. The first completion's terminal
EOS ID was `151645`; its 10-token count includes that EOS. It did not inherit
the second sequence's 32-token count.

The actual first response in both modes:

```text
Hello! How can I assist you today?
```

The actual second response in both modes, truncated by the requested cap:

```text
An inference server processes a request by following a series of steps that involve tokenization, prefill, decoding, the KV cache, and stopping. Here is a
```

The Hello response does not obey the one-word instruction, but both execution
paths agree. This is not evidence of a batching bug or improved answer quality.
Its prompt count here is 36; do not substitute the historical manual-loop
experiment's 28-token count.

### Verification and limitations

- Matched-library environment:
  `python -m unittest discover -s tests -p test_batched_generation.py -v`:
  **4 passed**.
- Project environment:
  `.venv/bin/python -m unittest discover -s tests -q`: **117 passed**.
- Real tiny-model checks cover unmodified greedy single/batch equivalence and
  later-call isolation. Controlled logits through the actual Transformers
  generation loop exercise early EOS, secondary EOS at the cap, unequal output
  lengths, reversed input order, and pad/EOS aliasing. Other checks cover
  exact-fit/overflow context budgets, invalid inputs, retained failure evidence,
  and overwrite refusal.
- The first focused test run exposed a synthetic fixture inconsistency: its
  secondary EOS was not marked special in the tokenizer. The fixture now marks
  both configured EOS IDs special, matching the intended Qwen setup. Its
  metadata subprocess check also explicitly disables tokenizer parallelism,
  avoiding the observed fork/thread-pool warning. Subsequent focused and full
  checks passed.
- The real CLI run above exercised cached Qwen, natural stopping, and the new
  batch path. Both saved results were compared against their individual
  references, including IDs, text, counts, and finish reasons.
- Invoking the CLI with the already-existing evidence path exited 2 before
  model loading. The evidence file's SHA-256 remained unchanged. This was an
  overwrite-protection check, not a repeated model run.

These observations establish correctness for these CPU cases, not universal
token equality across batch shapes, devices, or numerical implementations.
No latency, throughput, GPU memory, or speedup claim is supported by this run.
GPU batched execution has not been verified. The existing API still processes
one request at a time; this experiment does not batch incoming HTTP requests.

### Learning handoff and next step

The learner reports having completed the tokenizer exercise already. In this
conversation they correctly calculated batch shapes and padding counts and
identified the final padded position selected by `logits[:, -1, :]` after
clarification. Do not restart tokenization or basic padding drills.

Next, walk through `generate_batch()` using the saved 36/59-input and
10/32-output results: explain the common padded prefix, first generated EOS,
and why the short response is not counted as 32 tokens. Independent mastery
of this implementation remains to be demonstrated; passing tests do not prove it.

After that, obtain separate approval for a bounded GPU correctness check and
same-workload sequential versus batch-size 1/2/4 performance and memory study.
Keep static batching separate from future API scheduling or continuous batching.

## Stage 5: static batching benchmark

### Approved scope and implementation

On 2026-11-02, the learner approved implementing the controlled benchmark and
a small CPU smoke run. GPU/cloud execution still requires separate approval.
Added `benchmarks/batching.py`, `modal_batching.py`, and
`tests/test_benchmark_batching.py`. The API, scheduler, generation helpers,
existing warm benchmark, Modal entrypoint, and package environments are unchanged.

The benchmark loads one pinned float32 model and processes four fixed synthetic
conversations in input order. Batch size 1 calls existing `generate()` four
times; size 2 calls existing `generate_batch()` twice; size 4 calls it once.
There is no automatic request collection or HTTP scheduling change.

Each configuration is warmed independently with the complete four-prompt
workload. The first sequential warm-up supplies the output reference. Every
later returned completion records its input index, generated IDs, text, actual
counts, finish reason, and whether those IDs/counts/finish match the reference.
Differences remain available for investigation rather than being hidden.

`new_result()` validates bounds before work: output cap 1-128, warm-ups 1-2,
repeats 1-5. Defaults are 32 tokens, one warm-up, three measured workloads per
configuration. Configuration order is fixed at 1, 2, 4, not randomized.

### Timing, memory, and failure semantics

- Per-call timers bracket the complete generation helper, including formatting,
  tokenization, input/output transfers, prefill, decode, and text decoding.
  CUDA is synchronized before and after each call. CPU execution is synchronous.
- The workload timer covers all four prompts and includes call-timing overhead,
  dispatch, and loop bookkeeping. A preceding CUDA fence excludes earlier work.
  Model loading, warm-ups, JSON serialization, and reference comparisons are
  excluded. Workload time is not the sum of independently summarized medians.
- Aggregate throughput uses actual tokens divided by summed complete measured
  workload seconds. It does not average individual call rates or divide batch
  duration by batch size to claim request latency.
- Per-call durations are retained. A size-1 call contains one prompt, a size-4
  call contains four; their medians are different units of work, not equivalent
  per-user response latency. There is no HTTP/client timing, TTFT, or TPOT here.
- CUDA peak allocated and reserved bytes are recorded separately after resetting
  peak counters before each measured workload. These absolute PyTorch allocator
  peaks include the loaded model and exclude non-PyTorch allocations. Cached
  reservations survive across configurations; no `empty_cache()` is performed.
  CPU memory is intentionally unmeasured and represented by null.
- Failures and handled interrupts stop further generation. Already-completed
  calls retain outputs and durations, failed calls have no invented usage, and
  later calls are marked `not_run`. Incomplete workloads do not enter throughput
  or median summaries. Error text is not saved, only its type.
- Output files are exclusive. Remote transport failure before delivery leaves
  summaries null and `remote_results_received=false`; absence of local rows
  does not establish absence of remote work. Forced termination cannot save
  local evidence, and local interruption does not prove remote termination.

### Actual CPU smoke

Executed:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.batching \
  --device cpu --max-new-tokens 16 --warmups 1 --repeats 2 \
  --output benchmark-results/batching/cpu-benchmark-001.json
```

Recorded at `2026-11-02T19:19:16.122806+00:00`, status `completed`. The complete
JSON remains an ignored local artifact, including every measured call and output.

Runtime: Apple M1 Pro CPU, arm64, eight logical CPUs, six Torch intra-op and
eight inter-op threads, Python 3.12.13, float32. Core versions: Torch 2.6.0,
Transformers 4.57.6, NumPy 2.2.6, Hugging Face Hub 0.36.0, tokenizers 0.22.1,
safetensors 0.6.2. Model: Qwen/Qwen2.5-0.5B-Instruct, revision
`7ae557604adf67be50417f59c2c2f167def9a775`, template hash
`cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
Context limit 2048, temperature 0, top-p 1, cap 16, natural EOS enabled.

| Input | Prompt tokens | Output tokens per workload | Finish |
|---|---:|---:|---|
| Reply with exactly one word: Hello | 36 | 10 | stop |
| Explain inference-server request processing | 59 | 16 | length |
| What is 2 + 2? Reply with just the number. | 43 | 2 | stop |
| Explain in two sentences why attention uses a KV cache during autoregressive decoding. | 46 | 16 | length |

All three configurations completed one warm-up and two measured workloads.
Each workload generated 44 actual output tokens. Each configuration therefore
has 88 measured output tokens. All returned outputs matched the sequential
reference, including warm-ups after the reference; zero failures or mismatches.
There were nine total workloads and 21 generation calls including warm-ups.

| Batch size | Workload 0 (ms) | Workload 1 (ms) | Median workload (ms) | Median call (ms) | Summed measured ms | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 4728.203875 | 4238.533041 | 4483.368458 | 1199.982584 | 8966.736916 | 9.814050 |
| 2 | 4144.960209 | 4164.150375 | 4154.555292 | 2072.477688 | 8309.110584 | 10.590785 |
| 4 | 2209.768667 | 2176.515708 | 2193.142187 | 2193.139479 | 4386.284375 | 20.062539 |

Per-call measured durations in input/group order:

| Batch size | Workload | Call durations (ms) |
|---|---:|---|
| 1 | 0 | 833.406833, 1534.538334, 577.301416, 1782.936667 |
| 1 | 1 | 959.117583, 1440.847584, 242.294750, 1596.260917 |
| 2 | 0 | 2059.231125, 2085.724250 |
| 2 | 1 | 1989.715041, 2174.430209 |
| 4 | 0 | 2209.765542 |
| 4 | 1 | 2176.513416 |

Observed throughput ratios versus the sequential configuration, calculated
from unrounded values: **1.079145x** for size 2 and **2.044267x** for size 4.
GPU allocated/reserved memory fields are null because this was a CPU run.

This is a small validation benchmark, not a stable performance characterization.
It has two measured workloads per configuration, short outputs, fixed order,
and uncontrolled host conditions. Do not infer a stable p95/SLO, universal
batching speedup, or GPU behavior. Larger batches finish more work per call;
their call durations are not individual request latencies. These four-prompt,
16-cap results must not be compared directly with the old single-prompt 59/128
CPU/T4 measurements or the planned GPU run with a different output cap.

### Verification

- Matched-library environment: four new benchmark tests passed.
- Project environment: **121 tests passed** via
  `.venv/bin/python -m unittest discover -s tests -q`.
- New checks exercise actual tiny-model CPU generation and grouping, actual-token
  accounting, CPU avoidance of CUDA, parameter bounds, and unsupported-device
  rejection. Deterministic simulated CUDA work verifies leading/trailing fence
  boundaries, whole-workload versus call timing, reset peak allocation and
  retained reservations, and preservation of changed output IDs.
- Failure and interruption checks preserve completed calls but exclude partial
  workloads from throughput and median summaries; later configurations do not run.
- The real offline Qwen CLI run above completed and saved evidence. A subsequent
  invocation against that existing output path exited 2 with its SHA-256 unchanged;
  this protection check did not repeat generation.
- `modal run modal_batching.py --help` succeeded. Calling the decorated remote
  function only through `.local()` on the actual CPU host rejected execution
  with `This experiment requires CUDA.` before model loading.
- No remote invocation, GPU allocation, or image build was launched. Simulated
  timing tests do not establish actual GPU correctness or memory consumption.

### Original GPU proposal (superseded by the learner's 16-token run below)

```bash
~/.venvs/modal/bin/modal run modal_batching.py \
  --max-new-tokens 32 --warmups 1 --repeats 3 \
  --output benchmark-results/batching/t4-benchmark-001.json
```

The separate app reuses the existing pinned image/download layer and mounts the
three additional modules afterward. Requested limits: T4, 4096 MiB host RAM,
max one container, min zero containers, retries zero, 300-second function timeout;
the inherited CPU download step has a 600-second timeout. CUDA/float32 guards
are mandatory. Host RAM is not a GPU VRAM limit.

At these arguments: 28 generation calls, 48 completions including warm-ups,
and at most 32 new tokens per completion. First sequential warm-up establishes
the device-local reference; each batched warm-up compares outputs before that
configuration's measured workloads. Mismatches remain visible in results and
must qualify any performance interpretation.

These are not hard spending caps. GPU model loading, warm-ups, measurements,
and remaining container lifetime consume resources; image preparation may also
consume resources. Check credits and current Modal pricing before approval,
and stop the app if abandoning it. This 32-token proposal was not executed;
the learner later ran the 16-token variant recorded below.

Next learning task: interpret the workload/call-duration distinction and the
observed CPU throughput change without claiming that every user's latency
improved. GPU execution and any additional experiments require separate approval.

### First T4 batching result: 2026-11-04

#### Evidence source and scope

The learner ran the prepared benchmark and supplied successful Modal output,
including `status=completed`, `remote_results_received=true`, the saved path,
and `App completed`. Run:
https://modal.com/apps/lenando0307/main/ap-JXmGJQET81U3bwbbZPH2YC

Actual command:

```bash
~/.venvs/modal/bin/modal run modal_batching.py \
  --max-new-tokens 16 --warmups 1 --repeats 3 \
  --output benchmark-results/batching/t4-benchmark-001.json
```

The cap is 16, not the earlier proposed 32. Recorded UTC timestamp:
`2026-11-04T00:45:23.031925+00:00`. The existing JSON was read for arithmetic;
no model execution, GPU confirmation rerun, or test rerun was performed.
The raw artifact remains local and ignored, unchanged by this documentation update.

#### Configuration and correctness

- Tesla T4, device `cuda:0`, `torch.float32`, CUDA 12.4.
- Qwen/Qwen2.5-0.5B-Instruct, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`, template SHA-256
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Python 3.12.10; Torch 2.6.0+cu124, Transformers 4.57.6, NumPy 2.2.6,
  Hugging Face Hub 0.36.0, tokenizers 0.22.1, safetensors 0.6.2.
- Recorded host: Linux/gVisor x86_64, 17 logical CPUs, CPU model unavailable;
  one Torch intra-op thread and 17 inter-op threads.
- Context limit 2048, greedy temperature 0/top-p 1, maximum 16 new tokens,
  natural EOS, one active generation call at a time.
- Fixed configuration order 1, 2, 4; one full-workload warm-up then three measured
  workloads for each configuration. Twelve workloads and 28 generation calls
  including warm-ups, covering 48 completions.

All configurations completed. Each measured workload produced 44 actual tokens;
each configuration accumulated 132 measured tokens. There were zero warm-up
failures, measured failures, or output mismatches.

| Input | Prompt tokens | Completion tokens | Finish |
|---|---:|---:|---|
| Hello | 36 | 10 | stop |
| Inference-server explanation | 59 | 16 | length |
| 2 + 2 | 43 | 2 | stop |
| KV-cache explanation | 46 | 16 | length |

All later warm-ups and measurements matched the first GPU sequential warm-up's
generated IDs, prompt counts, and finish reasons. Reference rows have null
`matches_reference` because they establish that reference, not because they failed.
The CPU and GPU reference ID lists also match in the saved files. Agreement
establishes behavior for these cases, not universal numerical or answer-quality
equivalence. The Hello answer still exceeds the requested one word.

#### GPU timing and throughput

| Batch size | Workload 0 (ms) | Workload 1 (ms) | Workload 2 (ms) | Median workload (ms) | Median call (ms) | Sum measured ms | Output tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1088.988430 | 1054.301114 | 1062.624613 | 1062.624613 | 318.361215 | 3205.914157 | 41.173903 |
| 2 | 947.575302 | 927.173034 | 909.485786 | 927.173034 | 454.790147 | 2784.234122 | 47.409806 |
| 4 | 465.153640 | 470.446638 | 464.804698 | 465.153640 | 465.139004 | 1400.404976 | 94.258448 |

Calculated from the unrounded saved values:

- Size 2 / size 1 aggregate throughput: **1.151453x**.
- Size 4 / size 1 aggregate throughput: **2.289276x**.
- Sequential median / size-4 median workload duration: **2.284459x**,
  or **56.225968% lower median duration** for all four prompts.
- Median ratios and throughput ratios differ slightly because throughput uses
  summed measured time, not the median.

Size 2 requires two batch calls per workload, with a 454.79 ms median call
duration; size 4 requires one, with a 465.14 ms median call duration. This
observed call-level difference helps explain the aggregate result. It does
not identify how much came from weight reuse, numerical kernels, memory
bandwidth, or host/launch overhead; no profiler established those contributions.

These timers retain the documented whole-helper/whole-workload boundaries,
including CPU-side work and synchronized CUDA work. They exclude loading,
warm-ups, network invocation, result serialization, and comparisons. They
are not GPU-kernel-only, HTTP, TTFT, TPOT, or per-sequence delivery measurements.
Do not divide a batch call's duration by four and label it request latency.

#### GPU memory

| Batch size | Peak allocated bytes | Allocated MiB | Peak reserved bytes | Reserved MiB |
|---|---:|---:|---:|---:|
| 1 | 1992028672 | 1899.746582 | 2122317824 | 2024 |
| 2 | 1996925440 | 1904.416504 | 2122317824 | 2024 |
| 4 | 2008476160 | 1915.432129 | 2149580800 | 2050 |

MiB uses 1,048,576 bytes. The recorded peaks were identical across the three
measured workloads within each configuration.

Size 4 increased peak allocated memory by **16,447,488 bytes**, or
**15.685547 MiB (0.825665%)**, relative to size 1. Reserved memory increased
by **26 MiB**. Model weights are reused rather than copied for every batch row;
per-sequence state and intermediate tensors can grow. The observed difference
is not a measurement of KV-cache bytes alone.

Allocated describes PyTorch tensor allocation; reserved describes its larger
allocator pool. Peak counters were reset before each measured workload, but
cached reservations from previous configurations were not flushed. These
figures include model weights and exclude non-PyTorch allocations. They are
not complete device occupancy or a proof that longer contexts/bigger batches
will have similarly small memory growth.

#### Qualified comparison with the saved CPU smoke

The saved CPU and GPU files use identical conversations, cap 16, greedy settings,
revision, template hash, precision, and actual token counts. Six core package
release versions match after accounting for the CUDA Torch build suffix.

| Batch size | CPU median workload (ms) | GPU median workload (ms) | CPU output tokens/s | GPU output tokens/s | GPU/CPU throughput |
|---|---:|---:|---:|---:|---:|
| 1 | 4483.368458 | 1062.624613 | 9.814050 | 41.173903 | 4.195404x |
| 2 | 4154.555292 | 927.173034 | 10.590785 | 47.409806 | 4.476515x |
| 4 | 2193.142187 | 465.153640 | 20.062539 | 94.258448 | 4.698231x |

These are cross-environment observations, not hardware-isolated causal effects.
CPU had two measured workloads per configuration, GPU three. Python patch,
Torch build, OS, architecture, host CPU, thread settings (CPU 6/8; GPU host 1/17),
and execution conditions differ. Fixed order and short workloads further limit
generalization. The within-T4 batching comparison is the primary result.

#### Completed checkpoint and next lesson

Real CPU and GPU static batching are now exercised for the fixed four-prompt
workload. Performance and GPU allocator memory are recorded, with no observed
output discrepancy. The API still uses its original sequential worker; static
benchmark batches are not dynamic or continuous HTTP request batching.

Next, have the learner explain why higher aggregate throughput does not prove
every individual request has lower latency, and why reserved memory can stay
unchanged while allocated memory grows. Do not mark independent mastery from
successful execution alone. Three measurements do not establish a stable p95,
production SLO, universal speedup, larger-context memory behavior, or cloud cost.
No additional run is needed merely to confirm this supplied result.

## Stage 6: dynamic KV-cache scaling

### Implementation and scope

The learner approved implementation after inspection and a file-level proposal.
The existing `src/cached_generation.py` loop is now callable and import-safe;
it was not replaced with a second decoder. Batch rows stop independently,
left-padding masks determine rotary positions, and cache snapshots preserve
the prefill state rather than retaining a mutable cache reference.
The original standalone demonstration still runs through `main()`.

`benchmarks/cache_scaling.py` loads the pinned model once and runs five cases.
Each uses one discarded prefill warm-up, one manual generation, and one
Hugging Face reference through the existing `generate_batch` helper. Existing
metadata and exclusive JSON-writing helpers are reused. Errors/interruption
retain completed case data; a mismatch is a failed experiment, not a pass.
The API, scheduler, streaming code, and prior batching benchmarks are unchanged.
No dependencies or cloud jobs were added.

### Recorded CPU experiment: 2026-11-04

Local evidence: `benchmark-results/cache-scaling/cpu-dynamic-001.json`.
The record timestamp is `2026-11-04T04:35:03.304182+00:00`, not an exact
completion timestamp. Status: `completed`, with five successful cases.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.cache_scaling \
  --max-new-tokens 16 \
  --output benchmark-results/cache-scaling/cpu-dynamic-001.json
```

The file already exists; choose a fresh path for any separately approved repeat.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Chat-template SHA-256:
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- CPU float32; Apple M1 Pro, arm64, eight logical CPUs.
- Recorded platform: `macOS-26.6.2-arm64-arm-64bit`.
- Python 3.12.13; Torch 2.6.0; Transformers 4.57.6; NumPy 2.2.6;
  huggingface-hub 0.36.0; tokenizers 0.22.1; safetensors 0.6.2.
- Six intra-op threads, eight inter-op threads; context limit 2048.
- Greedy decoding, natural EOS stopping, output cap 16.
- DynamicCache; 24 layers, 14 query heads, 2 KV heads, head dimension 64,
  float32 cache elements of four bytes.

The prompt repeats `An inference server tokenizes prompts and retains attention
keys and values. ` either 1, 8, or 32 times, followed by a newline and
`Explain how the cache grows during decoding.` Chat formatting produces 52, 143,
and 455 input tokens. Multi-row cases repeat the same 143-token prompt, without
padding or prefix-cache sharing.

### Observed logical KV storage

Predicted and measured bytes matched exactly at both snapshots in every case.
K/V shapes are `[batch, 2, cached_positions, 64]` in each of 24 layers.

| Batch | Prompt positions | Final positions | Prefill bytes | Final bytes | Growth bytes |
|---:|---:|---:|---:|---:|---:|
| 1 | 52 | 67 | 1,277,952 | 1,646,592 | 368,640 |
| 1 | 143 | 158 | 3,514,368 | 3,883,008 | 368,640 |
| 1 | 455 | 470 | 11,182,080 | 11,550,720 | 368,640 |
| 2 | 143 | 158 | 7,028,736 | 7,766,016 | 737,280 |
| 4 | 143 | 158 | 14,057,472 | 15,532,032 | 1,474,560 |

| Batch | Prompt positions | Prefill MiB | Final MiB |
|---:|---:|---:|---:|
| 1 | 52 | 1.2187500 | 1.5703125 |
| 1 | 143 | 3.3515625 | 3.7031250 |
| 1 | 455 | 10.6640625 | 11.0156250 |
| 2 | 143 | 6.7031250 | 7.4062500 |
| 4 | 143 | 13.4062500 | 14.8125000 |

All nine manual completions selected 16 tokens and stopped at the length cap.
Each matched its same-batch Hugging Face reference exactly in token IDs, prompt
counts, and finish reason. There were no execution failures or mismatches.
Six checks per case passed, including prefill/final byte predictions and
`final_cached_positions = padded_prompt_width + longest_output_count - 1`.

The results demonstrate:

- Logical cache bytes scale with processed context positions at fixed batch.
- At the same 143/158 positions, batches 2 and 4 use exactly 2x and 4x the
  batch-1 cache bytes. Shared weights do not imply shared request caches.
- Every row adds 15 positions, not 16: the final selected token is not fed back.
  Growth is `15 * 24,576` bytes per row, or 360 KiB.
- Batch 4 adds 96 KiB per further processed position across its four rows.
  A budget equal to the current cache size leaves no room for continued growth.

### Diagnostic timing and limits

| Batch | Prompt positions | Prefill ms | Subsequent generation ms |
|---:|---:|---:|---:|
| 1 | 52 | 187.130250 | 2992.071166 |
| 1 | 143 | 248.643875 | 1261.868125 |
| 1 | 455 | 533.097833 | 1368.886917 |
| 2 | 143 | 329.665208 | 1955.046959 |
| 4 | 143 | 812.439625 | 2257.320542 |

These are single-run diagnostics, not a speed or throughput study. The prefill
timer covers the prompt forward; subsequent generation starts after selecting
token 1 and includes the remaining forwards, selections, and loop overhead.
Loading, tokenization, discarded prefill warm-up, cache inspection, serialization,
and reference generation are outside both timers. The decode path has no separate
warm-up, case order is fixed, and a separate import-safety probe ran while this
experiment was in progress. Do not infer performance scaling from this table.

Both GPU peak fields are `null`. The byte counts exclude model weights,
intermediates, allocator reservations, and process overhead; they are not a
measurement of physical allocation or total memory. The real-model cases use
equal-length rows and reach the cap. Mixed-length padding and unequal EOS
stopping are covered by tiny-model regressions, not claimed from this run.

### Verification and next lesson

- Matched CPU environment: all five new cache regression tests passed.
- Project environment: all 126 tests passed.
- Real tiny-Qwen checks cover unmodified greedy parity with left padding,
  independent EOS, secondary EOS at the cap, pad/EOS aliasing, no decode after
  all rows finish, GQA byte accounting, and retained dense rows.
- Runner checks cover five-case scaling, mismatch failure status, preserved
  measurements when reference generation fails, and overwrite refusal.
- A fresh-process import smoke blocked model/tokenizer loading and imported
  both new callable modules successfully.
- The preserved standalone command also ran in the project environment:
  `HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 .venv/bin/python src/cached_generation.py`.
  It observed 28 prompt positions, 10 generated tokens including EOS, 37 final
  cache positions, and exact Hugging Face token-ID parity. Output:
  `Hello! How can I assist you today?` Printed cache storage was 0.66 MiB
  after prefill and 0.87 MiB after generation; diagnostic times were 90.66 ms
  and 690.32 ms respectively. This separate smoke used the demonstration's
  existing unpinned loader and is not part of the pinned five-case result.

Next, ask the learner to explain the measured 15-position growth and the
batch-4 memory budget before comparing dynamic and static allocation.
GPU allocator measurements require separate approval. Paged allocation stays
conceptual; no custom allocator or continuous scheduler was implemented.
Collection-window microbatching remains deferred, not completed. Automated
verification does not establish independent learner mastery.

## CUDA cache-scaling support: prepared, not run

Historical implementation checkpoint: the completed T4 run below supersedes
this section's pending-run status. No GPU job ran during implementation itself.

### Approved implementation

The learner requested CUDA support after committing the CPU study as `f97c2e8`.
Approval covered implementation and local verification, not a paid GPU run.
No Modal app was launched, image built remotely, or GPU allocated during this work.

Changes:

- `src/cached_generation.py`: after each discarded prefill warm-up, release its
  outputs/cache, synchronize CUDA, and reset peak-memory counters. After manual
  generation and result preparation, synchronize and capture absolute allocated
  and reserved peaks. CPU/MPS results have null GPU peaks. The decode algorithm
  and logical-cache snapshots are unchanged.
- `benchmarks/cache_scaling.py`: CPU/CUDA selection, float32/device guards,
  no CPU fallback on CUDA failure, per-case peak persistence before HF reference
  generation, and lazy runtime imports so the Modal SDK environment needs no Torch.
- `modal_cache_scaling.py`: reuse the existing pinned image/model-download
  layer, mount only required source modules, validate bounds before submission,
  and preserve returned evidence or a local failed/interrupted transport record.
- `tests/test_cached_generation.py`: numerical CUDA-boundary regressions,
  CPU avoidance of CUDA APIs, unsupported-device/precision rejection, and
  unavailable-CUDA failure without loading a fallback model.

No `empty_cache()` is called. Peak resets do not release cached reservations.
Warm-ups and earlier cases, including their reference calls, can therefore
influence later reserved peaks. The current case's HF reference runs after its
peaks are captured. Peak allocated bytes include weights, cache, logits, and
temporary tensors; neither peak is a KV-only measurement or total-device usage.
The counters belong to the device, so measurements assume no concurrent CUDA work.

### Local verification

- Matched CPU environment: **7 cache tests passed**.
- Project environment: **128 tests passed**.
- The CUDA regression executes real tiny-Qwen CPU forwards under a simulated
  asynchronous allocator. It checks one-token and multi-token paths, warm-up
  tensor release, synchronization/reset effects, exclusion of reference
  allocations, and retained reservations across all five cases. These are
  measurement-boundary checks, not evidence of actual CUDA execution.
- `~/.venvs/modal/bin/python -m benchmarks.cache_scaling --help` exposes CPU/CUDA
  selection in the SDK-only environment.
- `~/.venvs/modal/bin/modal run modal_cache_scaling.py --help` loads the real
  Modal CLI without deployment. A fresh SDK-process import also verified that
  Torch was not imported.
- A throwaway local Modal-entrypoint smoke replaced remote submission with
  controlled failures. Invalid caps and existing output paths were rejected;
  failed/interrupted transport saved class-only evidence with
  `remote_results_received=false` and exit codes 1/130. No remote function
  was submitted. The throwaway files were removed.

### Real-Qwen CPU compatibility smoke

The new explicit CPU CLI path was exercised with a **two-token cap**, not by
repeating the original 16-token study:

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.cache_scaling \
  --device cpu --max-new-tokens 2 \
  --output benchmark-results/cache-scaling/cpu-cuda-support-smoke-001.json
```

Record timestamp: `2026-11-04T18:50:15.306432+00:00`; status `completed`.
The pinned model revision, chat template, float32 dtype, matched library versions,
Apple M1 Pro host, and 6/8 Torch thread settings match the earlier CPU study.
All five cases passed all six checks. All nine manual completions selected two
tokens and stopped at the cap; each matched its same-batch HF reference exactly.
GPU peak fields remained null.

| Batch | Prompt positions | Final positions | Prefill bytes | Final bytes |
|---:|---:|---:|---:|---:|
| 1 | 52 | 53 | 1,277,952 | 1,302,528 |
| 1 | 143 | 144 | 3,514,368 | 3,538,944 |
| 1 | 455 | 456 | 11,182,080 | 11,206,656 |
| 2 | 143 | 144 | 7,028,736 | 7,077,888 |
| 4 | 143 | 144 | 14,057,472 | 14,155,776 |

Predicted and measured KV bytes agreed at both snapshots. Each row grew by
one processed position, or 24 KiB. Diagnostic single-run timings are preserved
in the JSON, not treated as a performance comparison with the 16-token study.

### Pending GPU run and learning handoff

After separate approval, the intended command is:

```bash
~/.venvs/modal/bin/modal run modal_cache_scaling.py \
  --max-new-tokens 16 \
  --output benchmark-results/cache-scaling/t4-dynamic-001.json
```

Bounds: one T4, 4096 MiB host RAM, at most one container, zero minimum warm
containers, 300-second function timeout, no retries. The inherited CPU download
step has a 600-second timeout. The five fixed cases run one manual generation
and one HF reference each, plus one discarded prefill warm-up each.
The cap defaults to 16 and is bounded to 1..128.

These are resource/workload limits, not a hard spending cap. Loading, warm-ups,
image work, and container lifetime can incur cost outside measurement windows.
Check credits/pricing before approval. A local interruption does not establish
that remote work stopped; use Modal to stop the app if abandoning it.
No returned result means no received evidence, not proof of no remote work.

No actual CUDA peak values, T4 correctness, or GPU performance are claimed yet.
Compare logical bytes only at equal batch, cached positions, and dtype. GPU
outputs can differ from CPU; correctness compares manual and HF paths on the
same device. Dynamic/static comparison and paged-allocation concepts remain next.

The learner already explained why the last selected token is absent from the
cache and why continued decoding needs more cache space. Do not restart those
questions; use the eventual GPU results to distinguish logical KV storage,
allocated tensor memory, and reserved allocator memory.

## First T4 dynamic-cache result: 2026-11-04

### Supplied evidence and matching control

The learner ran the prepared command and supplied successful output.
The assistant read the saved JSON and compared it with the original 16-token
CPU study; no model computation, cloud job, or test suite was rerun.
Only README/evidence documentation changed while recording this result.

- GPU artifact: `benchmark-results/cache-scaling/t4-dynamic-001.json`.
- CPU control: `benchmark-results/cache-scaling/cpu-dynamic-001.json`.
- Record timestamp: `2026-11-04T18:56:43.964478+00:00`, not an exact
  completion timestamp.
- Status: `completed`; `remote_results_received=true`; all five cases successful.
- [Modal app completed](https://modal.com/apps/lenando0307/main/ap-3u0we2FS5gqQLYux4Deuy2).
- The two-token `cpu-cuda-support-smoke-001.json` is a compatibility smoke,
  not the matching output-length control for this GPU study.

```bash
~/.venvs/modal/bin/modal run modal_cache_scaling.py \
  --max-new-tokens 16 \
  --output benchmark-results/cache-scaling/t4-dynamic-001.json
```

The artifact already exists and must not be overwritten. No additional run is
needed just to confirm the supplied result; further cloud work needs approval.

### Runtime and correctness

- Tesla T4, `cuda:0`, float32, CUDA 12.4.
- `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Chat-template SHA-256:
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- Python 3.12.10; Torch 2.6.0+cu124; Transformers 4.57.6; NumPy 2.2.6;
  huggingface-hub 0.36.0; tokenizers 0.22.1; safetensors 0.6.2.
- Linux 4.19 gVisor x86_64, glibc 2.36; host CPU name unavailable;
  17 logical CPUs, one intra-op and 17 inter-op Torch threads.
- Greedy decoding, cap 16, context limit 2048, natural EOS enabled.
- DynamicCache, 24 layers, 14 query heads, 2 KV heads, head dimension 64,
  four bytes per float32 cache element.

Model revision, template, prompts, case order, dtype, and generation settings
match the CPU control. Every one of the nine manual completions selected 16
tokens and stopped at the length cap. All matched the same-device HF reference
in token IDs, prompt counts, and finish reasons. All six checks per case passed;
there were no execution failures or output mismatches.

Comparison of the saved files also found exact CPU/GPU generated-ID agreement
and identical prefill/final cache snapshots across all five cases. This does
not prove bitwise-equal logits or universal cross-device output equality.

### Logical KV storage and allocator peaks

| Batch | Prompt positions | Final positions | Prefill KV bytes | Final KV bytes | Peak allocated bytes | Peak reserved bytes |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 52 | 67 | 1,277,952 | 1,646,592 | 2,019,345,408 | 2,155,872,256 |
| 1 | 143 | 158 | 3,514,368 | 3,883,008 | 2,077,401,088 | 2,246,049,792 |
| 1 | 455 | 470 | 11,182,080 | 11,550,720 | 2,278,574,080 | 2,531,262,464 |
| 2 | 143 | 158 | 7,028,736 | 7,766,016 | 2,169,424,896 | 2,533,359,616 |
| 4 | 143 | 158 | 14,057,472 | 15,532,032 | 2,353,970,688 | 2,885,681,152 |

| Batch | Prompt positions | Final KV MiB | Peak allocated MiB | Peak reserved MiB |
|---:|---:|---:|---:|---:|
| 1 | 52 | 1.5703125 | 1925.797852 | 2056 |
| 1 | 143 | 3.7031250 | 1981.164063 | 2142 |
| 1 | 455 | 11.0156250 | 2173.017578 | 2414 |
| 2 | 143 | 7.4062500 | 2068.924805 | 2416 |
| 4 | 143 | 14.8125000 | 2244.921387 | 2752 |

The measured logical bytes equal the model-derived prediction at both
snapshots in every case. Each sequence adds 15 processed positions, or 360 KiB;
the final selected token is not processed. At 143 prompt/158 final positions,
logical KV storage doubles and quadruples with batch sizes 2 and 4.

The allocator peaks cover manual generation after warm-up output/cache release,
CUDA synchronization, and peak reset. Capture happens after a final synchronization
and before the current HF reference call. They include weights, KV, logits,
and temporary tensors, not just cache. No `empty_cache()` was used.

The 455-position batch-1 case was followed by the 143-position batch-2 case.
Peak allocated memory decreased by **104.092773 MiB**, while peak reserved
memory increased by only **2 MiB**, from 2414 to 2416 MiB. Reserved memory can
retain reusable allocator blocks from warm-ups and earlier manual/reference
runs rather than track each case's live tensor demand.

At 143 positions, moving from batch 1 to batch 4 increased final logical KV by
**11.109375 MiB**, but peak allocated memory increased by **263.757324 MiB**.
The latter is not a KV-cache measurement: it includes other live tensors and
can peak at a different point from the final cache snapshot. No allocation
profile was captured, so these numbers do not identify an exact breakdown.

The two allocator columns are independent maxima over a measurement interval,
not necessarily simultaneous values. Do not subtract final KV from peak
allocated to estimate weights, or subtract the two peaks to claim instantaneous
unused memory. Retained reservations alone do not demonstrate a memory leak.

### Diagnostic timings and limits

| Batch | Prompt positions | Prefill ms | Subsequent generation ms |
|---:|---:|---:|---:|
| 1 | 52 | 40.474080 | 463.175914 |
| 1 | 143 | 56.150981 | 454.473058 |
| 1 | 455 | 143.129555 | 572.223150 |
| 2 | 143 | 106.410666 | 589.019179 |
| 4 | 143 | 164.911402 | 791.882832 |

One observation per case, in fixed order, with prompt-only warm-up and no
separate decode warm-up. Timers exclude loading, tokenization, warm-up, cache
inspection, serialization, and HF reference generation. They are not HTTP
latency, full TTFT, a stable throughput benchmark, or a production SLO.
CPU/GPU runtime environments also differ in Python patch, Torch build,
OS/architecture, host CPU, and thread counts (CPU 6/8 versus GPU host 1/17).
The primary result is correctness and memory accounting, not a hardware-isolated
speedup claim. This run does not measure OOM capacity or billing totals.

### Completed checkpoint and next step

Real CPU and T4 dynamic-cache scaling are now demonstrated for the fixed
workloads. The API remains sequential; no static cache, paged allocator, or
continuous scheduler was introduced.

Before new code, distinguish the batch-4 final logical cache (14.8125 MiB),
peak allocated tensors (2244.92 MiB), and peak reserved allocator memory
(2752 MiB). Then inspect the installed static-cache interface and propose a
bounded dynamic/static comparison, separating occupied positions from allocated
capacity. Source changes and additional cloud execution need their own approval.

## CPU dynamic/static cache allocation: 2026-11-05

### Approved scope and implementation

The learner approved the CPU comparison after distinguishing KV-cache allocation
from request batching. This step changes cache allocation, not batching.
No GPU job, compilation, custom allocator, scheduler, dependency, or new source
module was added. Existing CPU/T4 dynamic artifacts remain untouched.

- `src/cached_generation.py`: reuse the same manual decoder with optional
  `cache_capacity`. None keeps dynamic allocation; an integer constructs the
  installed `StaticCache` for both discarded warm-up and a separate measured run.
  Validate capacity before forwarding, including the exact required bound
  `prompt_width + output_cap - 1`, because the final selected token is not processed.
- Cache writes use an explicit padded-slot cursor, separate from per-row rotary
  positions. Native Qwen masking excludes padding, future positions, and unused
  static capacity. No custom attention-mask implementation was introduced.
- Snapshots record processed `cached_positions`, diagnostic
  `reported_cache_length`, actual `allocated_positions`, formula-derived
  `predicted_used_bytes`, predicted backing-tensor bytes, and directly summed
  K/V tensor bytes. Used bytes are not an additional allocation.
- `benchmarks/cache_scaling.py --compare-static` selects three batch-1 cases on
  the existing medium prompt: dynamic, static 160, and static 256. The output
  cap is bounded to 1..16; CUDA selection is rejected before model loading.
  Each case retains HF parity checks, adds capacity checks, and compares IDs,
  prompt counts, and finish reasons with the current-run dynamic baseline.
- Default five-case dynamic CPU/CUDA studies and Modal commands remain
  available. Fresh-path persistence and partial-failure evidence are reused.

### Installed API finding

Both installed Transformers 4.57.6 and 5.18.0 support
`StaticCache(config=model.config, max_cache_len=capacity)`.
Their backing tensors are allocated lazily at first update, then changed in place.
The manual experiment stays eager; tagging static storage in the library is not
the same as compiling the model.

In 4.57.6, `StaticCache.get_seq_length()` counts nonzero key rows in the first
batch member/head. It can therefore undercount zero-valued padding or legitimate
zero keys. The newer implementation tracks cumulative length. An in-memory
two-row probe observed the older raw query return 1 after processing three
padded slots, and 2 after the next decode; the newer query returned 3 and 4.
Both had backing shape `[2, 2, 8, 4]`.

The decoder uses explicit write positions and counts successful processed slots
rather than using this content heuristic to choose its write cursor. It still
records the raw library length for transparency. In the real unpadded Qwen
comparison below, raw and tracked lengths agreed at both snapshots.

### Real-Qwen CPU experiment

Artifact: `benchmark-results/cache-scaling/cpu-allocation-001.json`.
Record timestamp: `2026-11-05T23:45:54.111237+00:00`, not exact completion time.
Status: `completed`; all three cases successful.

```bash
HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  ~/.venvs/inference-cpu-control/bin/python -m benchmarks.cache_scaling \
  --compare-static --device cpu --max-new-tokens 16 \
  --output benchmark-results/cache-scaling/cpu-allocation-001.json
```

The file exists; choose a fresh output path for any later run.

- Model: `Qwen/Qwen2.5-0.5B-Instruct`, revision
  `7ae557604adf67be50417f59c2c2f167def9a775`.
- Chat-template SHA-256:
  `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f`.
- CPU float32, Apple M1 Pro, arm64, eight logical CPUs;
  platform `macOS-26.6.2-arm64-arm-64bit`; six intra-op/eight inter-op threads.
- Python 3.12.13, Torch 2.6.0, Transformers 4.57.6, NumPy 2.2.6,
  huggingface-hub 0.36.0, tokenizers 0.22.1, safetensors 0.6.2.
- Batch 1, actual prompt length 143, context limit 2048, greedy output cap 16.
- 24 layers, 2 KV heads, head dimension 64, four-byte cache elements:
  24,576 bytes per slot across all layers.
- Fixed order: dynamic, static 160, static 256. Each has one discarded prefill
  warm-up, one manual generation, and one HF reference generation.

All cases selected the same 16 token IDs and stopped at the length cap.
The output begins `The cache grows during decoding in the context of an inference
server, which is a`. All matched the same-device HF reference. All eight checks
per case passed; no mismatch or execution failure occurred.

### Observed storage and occupied positions

| Cache | Prefill processed | Final processed | Prefill tensor capacity | Final tensor capacity | Prefill K/V bytes | Final K/V bytes |
|---|---:|---:|---:|---:|---:|---:|
| Dynamic | 143 | 158 | 143 | 158 | 3,514,368 | 3,883,008 |
| Static 160 | 143 | 158 | 160 | 160 | 3,932,160 | 3,932,160 |
| Static 256 | 143 | 158 | 256 | 256 | 6,291,456 | 6,291,456 |

| Cache | Prefill K/V MiB | Final K/V MiB | Final unused slots | Final unused logical bytes |
|---|---:|---:|---:|---:|
| Dynamic | 3.3515625 | 3.703125 | 0 | 0 |
| Static 160 | 3.7500000 | 3.750000 | 2 | 49,152 |
| Static 256 | 6.0000000 | 6.000000 | 98 | 2,408,448 |

Predicted backing bytes matched direct tensor sums in every snapshot. Useful
processed-slot storage was identical across all cases: 3,514,368 bytes at
prefill and 3,883,008 at completion. Static capacity did not turn unused slots
into valid history. Dynamic backing storage grew by 368,640 bytes; both static
backing sizes stayed constant while 15 additional positions were processed.

The 160-slot cache ends with 48 KiB of unused capacity. The 256-slot cache ends
with 2.296875 MiB unused. These are logical K/V tensor byte counts, not process
RSS or allocator reservations. Both GPU peak fields are null in this CPU study.

### Diagnostic timings and verification

| Cache | Prefill ms | Subsequent generation ms |
|---|---:|---:|
| Dynamic | 233.031209 | 1318.107625 |
| Static 160 | 231.150417 | 1417.145042 |
| Static 256 | 199.055208 | 1289.432750 |

Single-run diagnostics only: fixed case order, prompt-only warm-up, no dedicated
decode warm-up, no compiler, and no controlled performance repetitions. Prefill
includes first-forward cache initialization. Cache inspection, serialization,
and HF reference calls are outside the timers. Do not claim static is faster or
slower from this table; the experiment establishes memory behavior and parity.

- Matched CPU environment: **10 cache regression tests passed**.
- Project environment: **131 tests passed**, including the same static checks
  against the installed newer Transformers implementation.
- Regression coverage includes exact-fit/undersized capacity, invalid capacity
  types/limits, early EOS without filling the buffer, cache freshness, and
  unequal-length left-padded rows whose pad keys can be zero.
- A real tiny-model test deliberately fills unused static value slots with
  large nonzero values, then checks every forward's final-position logits
  against the dynamic baseline, as well as token IDs. This exercises actual
  attention masking, not a mocked decoder.
- The CPU comparison runner verifies constant static backing bytes against
  growing dynamic bytes. CUDA comparison requests are rejected.
- SDK-only CLI help displays `--compare-static` without importing/loading a model.

### Learning handoff

Static and dynamic cache allocation now produce matching outputs for the same
CPU workload, with the expected used-position versus backing-capacity difference.
This is not static versus continuous batching. The learner previously confused
those terms; keep request grouping and cache allocation distinct.

Next, interpret why both static cases have 158 used positions but different
footprints, and why fixed allocation alone does not guarantee a speedup.
Then introduce paged allocation conceptually: allocate blocks as requests grow,
without implementing a custom allocator. No additional GPU run is required to
establish the observed CPU allocation tradeoff, and new cloud work needs approval.
