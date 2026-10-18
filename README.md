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

From another terminal:

```bash
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
```

| Endpoint/state | Status | Body |
|---|---|---|
| `/health`, serving HTTP | 200 | `{"status":"alive"}` |
| `/ready`, initialized and idle | 200 | `{"ready":true,"status":"ready"}` |
| `/ready`, admission guard occupied | 503 | `{"ready":false,"status":"busy"}` |
| `/ready`, uninitialized, if reachable | 503 | `{"ready":false,"status":"not_ready"}` |
| `/ready`, shutting down, if reachable | 503 | `{"ready":false,"status":"shutting_down"}` |

Readiness is a snapshot, not a reservation. Uvicorn may refuse connections until
startup completes; connection refusal is not an HTTP 503 response.

Press Ctrl-C to stop normally. Shutdown marks the application unavailable,
waits for executor work off the event loop, then releases the runtime. It does
not cancel native model computation; a permanently stuck worker can prevent
graceful shutdown.

Only health/readiness routes are implemented for the local service.
`POST /v1/chat/completions`, generation admission, HTTP error mapping, and SSE
remain later checkpoints. The executor and guard exist, but no HTTP generation
request can be submitted yet.

## Run behavioral tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers request validation, context budgets, and lifecycle transitions.
It uses offline inference fixtures and controlled lifecycle resources; `httpx`
is the HTTP test client. Real-Qwen smoke evidence is recorded separately in
`logs/practice.md`.

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
