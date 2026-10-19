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
| 429 | `server_busy` | Another producer owns inference; no waiting queue |
| 503 | `not_ready` | Application unavailable or shutting down |
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

The internal execution path is `submit_inference(app, validated_request)` in
`src/api.py`. Call it on the running application's event loop with an already
validated `ChatCompletionRequest`, then await the returned future. It rejects
unavailable state with `InferenceUnavailable` and an occupied slot with
`InferenceBusy`; excess work is not queued.

The dedicated worker performs formatting, tokenization, context validation, and
generation, releasing the guard only when its work exits. The returned result
is already shielded: canceling its consumer does not cancel the retained producer
or free the slot early. Completion callbacks observe errors even when delivery
is abandoned; they do not log raw exception text. Shutdown drains producer
completion before clearing the runtime.

For both response modes, a detected disconnect ends delivery, not inference.
The finite producer keeps running under the requested cap, and new requests
still receive 429 until it exits. Undelivered output is discarded. A disconnected
client is not promised a terminal event or `[DONE]`. Canceling a Python future
cannot safely kill a stuck native call: inference remains occupied, and process
restart is required rather than launching a replacement over the same model.

The library streamer queue is not intrinsically bounded. This stage permits
only one producer and at most 512 generated tokens; explicit slow-consumer
backpressure and cooperative stopping remain later work. Real TCP disconnects
and controlled forward failures have been verified with Qwen, alongside
deterministic ASGI ownership regressions.

## Run behavioral tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers request validation, context budgets, lifecycle transitions,
busy admission, consumer cancellation, SSE progress and terminal metadata,
preparation/producer failures, and disconnect ownership. It uses offline
inference fixtures and controlled worker resources. HTTPX covers ordinary
requests; direct ASGI connections expose streaming frames without buffering.
Real-Qwen HTTP smoke evidence is recorded separately in `logs/practice.md`.

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
