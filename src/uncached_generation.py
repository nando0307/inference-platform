from time import perf_counter

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

print("Decoding method:")
print("1. Greedy")
print("2. Sampling")

while True:
    choice = input("Choose 1 or 2: ").strip()

    if choice in {"1", "2"}:
        break

    print("Please enter 1 or 2.")

temperature = 0.7

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
tokens = tokenizer.tokenize(prompt)
print(f"tokens for the prompt: {tokens}")
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"ids: {ids}")
decoded_string = tokenizer.decode(ids)
print(f"decoded_string {decoded_string}")

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

generated_ids = []
max_new_tokens = 100

eos_token_ids = model.generation_config.eos_token_id
if eos_token_ids is None:
    eos_token_ids = []
elif isinstance(eos_token_ids, int):
    eos_token_ids = [eos_token_ids]

with torch.inference_mode():
    # Warm up the prompt-forward path, then discard its outputs.
    warmup_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    del warmup_outputs

    prefill_start = perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    prefill_seconds = perf_counter() - prefill_start

    print("Input IDs shape:", input_ids.shape)
    print("Logits shape:", outputs.logits.shape)

    decode_start = None
    decode_seconds = 0.0
    stop_reason = "maximum new tokens"

    for step in range(max_new_tokens):
        # Select from prefill logits first, then from the previous forward pass.
        next_token_logits = outputs.logits[:, -1, :]  # [1, V]
        if choice == "1":
            # Greedy
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )
        else:
            # Sampling
            probabilities = torch.softmax(
                next_token_logits / temperature,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

        token_id = next_token.item()
        generated_ids.append(token_id)
        del next_token_logits

        if token_id in eos_token_ids:
            stop_reason = "end-of-sequence token"
            break
        if step + 1 == max_new_tokens:
            break

        # Match the cached timer: start after selecting token 1.
        if step == 0:
            decode_start = perf_counter()

        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)],
            dim=1,
        )

        # Unlike cached decoding, recompute the entire growing sequence.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    if decode_start is not None:
        decode_seconds = perf_counter() - decode_start

print(f"Prompt forward-pass time: {prefill_seconds * 1000:.2f} ms")
print(f"Subsequent generation time: {decode_seconds * 1000:.2f} ms")
print(f"Tokens generated after the first: {max(0, len(generated_ids) - 1)}")
print(f"Stopped: {stop_reason}.")
print("\nResponse:")
print(tokenizer.decode(generated_ids, skip_special_tokens=True))
