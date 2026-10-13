import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from time import perf_counter

print("Loading Model... ")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Hello world! Which paper mark the turning point for AI research?"

messages = [
    {"role": "system", "content": "You are Qwen, a helpful assistant"},
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

input_ids = model_inputs["input_ids"]
attention_mask = model_inputs["attention_mask"]

max_new_tokens = 100
generated_ids = []

eos_token_ids = model.generation_config.eos_token_id
if eos_token_ids is None:
    eos_token_ids = []
elif isinstance(eos_token_ids, int):
    eos_token_ids = [eos_token_ids]

with torch.inference_mode():
    # Warm up the prompt-forward path. Do not reuse this run's cache.
    warmup_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    del warmup_outputs
    # Prefill: process the entire prompt once.
    prefill_start = perf_counter()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )

    prefill_seconds = perf_counter() - prefill_start
    past_key_values = outputs.past_key_values

    print("Input IDs shape:", input_ids.shape)
    print("Logits shape:", outputs.logits.shape)
    print("Cache type:", type(past_key_values).__name__)
    print("Cached sequence length:", past_key_values.get_seq_length())

    decode_start = None
    decode_seconds = 0.0
    stop_reason = "maximum new tokens"

    for step in range(max_new_tokens):
        # Select from prefill logits first, then from the previous decode step.
        next_token = torch.argmax(
            outputs.logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
        token_id = next_token.item()
        generated_ids.append(token_id)

        # Selection does not grow the cache. Stop before an unnecessary forward pass.
        if token_id in eos_token_ids:
            stop_reason = "end-of-sequence token"
            break
        if step + 1 == max_new_tokens:
            break

        # Start after selecting token 1, only if another token is needed.
        if step == 0:
            decode_start = perf_counter()

        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)],
            dim=1,
        )

        # Decode: process only the selected token, reusing the previous K/V state.
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values

    if decode_start is not None:
        decode_seconds = perf_counter() - decode_start

print(f"Prompt forward-pass time: {prefill_seconds * 1000:.2f} ms")
print(f"Subsequent generation time: {decode_seconds * 1000:.2f} ms")
print(f"Tokens generated after the first: {max(0, len(generated_ids) - 1)}")
print(f"Final cache length: {past_key_values.get_seq_length()}")
print(f"Stopped: {stop_reason}.")
print("\nResponse:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
