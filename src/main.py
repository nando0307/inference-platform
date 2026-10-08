import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
input_ids = model_inputs["input_ids"]
embedding_layer = model.get_input_embeddings()
with torch.inference_mode():
    embeddings = embedding_layer(input_ids)

print("\nToken IDs shape:", input_ids.shape)
print("Embedding table shape:", embedding_layer.weight.shape)
print("Input embeddings shape:", embeddings.shape)


# Inspect Qwen's actual attention dimensions
config = model.config

print("\nHidden dimension:", config.hidden_size)
print("Query heads:", config.num_attention_heads)
print("Key/value heads:", config.num_key_value_heads)
print("Transformer layers:", config.num_hidden_layers)


with torch.inference_mode():
    outputs = model(**model_inputs, use_cache=False)

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

next_token_id = torch.argmax(next_token_logits, dim=-1).item()

print("Chosen token ID:", next_token_id)
print("Chosen token text:", repr(tokenizer.decode([next_token_id])))

# original input
input_ids = model_inputs["input_ids"]
# match the next token with input id's dtype and device
next_token_tensor = input_ids.new_tensor([[next_token_id]])

extended_ids = torch.cat([input_ids, next_token_tensor], dim=1)
extended_mask = torch.cat(
    [model_inputs["attention_mask"], torch.ones_like(next_token_tensor)],
    dim=1,
)
print("\nBefore append:", input_ids.shape)
print("After append:", extended_ids.shape)

with torch.inference_mode():
    second_outputs = model(
        input_ids=extended_ids,
        attention_mask=extended_mask,
        use_cache=False,
    )

second_token_logits = second_outputs.logits[0, -1, :]
second_token_id = torch.argmax(second_token_logits).item()

print("Second forward-pass logits:", second_outputs.logits.shape)
print("Second token ID:", second_token_id)
print("Second token text:", repr(tokenizer.decode([second_token_id])))

print(
    "First two generated tokens:",
    repr(tokenizer.decode([next_token_id, second_token_id])),
)


# generated_ids = [
#     output_ids[len(input_ids) :]
#     for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("\nResponse:")
# print(response)
