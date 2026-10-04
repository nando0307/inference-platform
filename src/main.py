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

prompt = "Hello world! Which paper mark the turning point for AI research?"
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

with torch.inference_mode():
    outputs = model(**model_inputs)

logits = outputs.logits
prob = torch.softmax(logits[0, -1, :], dim=-1)
print(
    prob.shape,
    prob.sum().item(),
    prob.min().item(),
    prob.max().item(),
)
top_5 = torch.topk(prob, k=5)
token_ids = top_5.indices.tolist()
probabilities = top_5.values.tolist()
token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
print("\nToken ID | Token representation | Probability")

for token_id, piece, probability in zip(token_ids, token_pieces, probabilities):
    print(f"{token_id} | {piece!r} | {probability:.6f}")

next_token_logits = logits[0, -1, :]
print("Next-token logits shape:", next_token_logits.shape)
print("First 10 vocabulary scores:", next_token_logits[:10])

logits = outputs.logits

# generated_ids = [
#     output_ids[len(input_ids) :]
#     for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("\nResponse:")
# print(response)
