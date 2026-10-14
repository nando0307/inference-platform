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

    first_layer = past_key_values.layers[0]

    print("First-layer keys shape:", first_layer.keys.shape)
    print("First-layer values shape:", first_layer.values.shape)
    print("Cache dtype:", first_layer.keys.dtype)
    print("Bytes per element:", first_layer.keys.element_size())
    print("Cached layers:", len(past_key_values.layers))

    prefill_cache_bytes = 0

    for layer in past_key_values.layers:
        prefill_cache_bytes += layer.keys.numel() * layer.keys.element_size()
        prefill_cache_bytes += layer.values.numel() * layer.values.element_size()

    print(f"KV storage after prefill: {prefill_cache_bytes / 1024**2:.2f} MiB")

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
final_cache_bytes = 0

for layer in past_key_values.layers:
    final_cache_bytes += layer.keys.numel() * layer.keys.element_size()
    final_cache_bytes += layer.values.numel() * layer.values.element_size()

print(f"KV storage after generation: {final_cache_bytes / 1024**2:.2f} MiB")
