import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

print("Loading Model... ")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Hello world! Which paper mark the turning point for AI research? Write it in 25 words."
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

prompt_length = input_ids.shape[1]
max_new_tokens = 1000

eos_token_ids = model.generation_config.eos_token_id
if eos_token_ids is None:
    eos_token_ids = []
elif isinstance(eos_token_ids, int):
    eos_token_ids = [eos_token_ids]

with torch.inference_mode():
    for step in range(max_new_tokens):
        # 1. Process the entire current sequence
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # 2. Get next-token scores: [1, S, V] -> [1,V]
        next_token_logits = outputs.logits[:, -1, :]

        # 3. Choose one token, retaining shape [1,1]
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        # 4. Append the chosen token and its mask entry
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)],
            dim=1,
        )

        token_id = next_token.item()

        print(
            f"Step {step + 1}: "
            f"length={input_ids.shape[1]}, "
            f"id={token_id}, "
            f"text={tokenizer.decode([token_id])!r}"
        )

        # 5. Stop if the model chose an end-of-sequence token.
        if token_id in eos_token_ids:
            print("Stopped: end-of-sequence token.")
            break

generated_ids = input_ids[0, prompt_length:]

print("\nResponse:")
print(tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True))
