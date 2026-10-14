```text
# inference platform
Hi i am Do, i am learning about the inference !
my dream is to becoming a llm engineer xD
```


# Session handoff: 2026-10-13

## Start here, next mentor

The learner has completed manual cached and uncached generation, single-run
timing comparisons with 34-token and 516-token prompts, and direct inspection
of KV tensor storage after prefill and generation. Do not restart at installing
Transformers, `model.generate()`, tokenization, or selecting one token.

This is Stage 1's basic KV-cache/prefill/decode work, extended with introductory
timing and memory experiments. It is not completion of the full benchmarking,
cache-scaling, or serving roadmap.
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
- Do not jump to FastAPI, vLLM, GPU optimization, or a larger architecture yet.
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
  separate prompt/subsequent-generation timers.
- `src/cached_generation.py`: greedy generation; full-prompt prefill followed
  by single-token forwards passing `past_key_values`, with a full growing
  attention mask. Includes matching warm-up and timing boundaries, first-layer
  K/V shape/dtype inspection, and total KV tensor bytes before and after generation.
- Both generation scripts use a 100-new-token limit and the model's configured
  EOS IDs, decode the generated IDs together, and print outside the timed loop.
- Both generation scripts currently contain the same 516-token long prompt.
  The 34-token prompt below is a historical baseline, not the current input.

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
unlike the short-prompt comparison, no programmatic long-prompt token-ID equality
check was recorded.

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

## Repository checkpoints and documentation

Recent commits inspected while updating this log:

| Commit | Date | Checkpoint |
|---|---|---|
| `e18ed80` | 2026-10-12 | Cached greedy generation, uncached script rename, script documentation |
| `11ccaba` | 2026-10-13 | Cached/uncached latency measurement and documentation |
| `c75022f` | 2026-10-13 | KV-cache timing results and session checkpoint |

The current files include the longer prompts and memory inspection. The latest
listed commit does not establish that these newer changes have been committed.
No commit was created as part of this log update.

Documentation roles: README is the entry point and run guide; this file holds
experiment evidence and the live learning handoff; `plan.md` is the roadmap.
Keep detailed measurements here rather than copying them into every document.

## Exact stopping point and how to resume

The prompt-length experiment and basic K/V memory inspection are complete.
The latest checkpoint is the learner's observed 12.09 MiB after prefill and
14.41 MiB after generation at cache lengths 516 and 615. Do not repeat those
experiments merely because an older roadmap section still calls them proposed.

No next coding experiment has been agreed. Resume with a short review of
compute saved versus memory retained, then agree on one bounded Stage 1 task.
A reasonable remaining task is a matched greedy comparison against Hugging
Face `generate()`; it is a proposal, not an implemented or approved change.
Ask before editing files, or provide a small exercise for the learner to write.

Other roadmap work not established by this session: `generate()` parity,
top-k/top-p sampling, a controlled temperature study, CPU/MPS comparison,
repeated statistical benchmarks, total/peak process-memory profiling, actual
concurrency/batching experiments, and any HTTP server. Do not mark all of
Stage 1 or the deeper cache-scaling phase complete.

