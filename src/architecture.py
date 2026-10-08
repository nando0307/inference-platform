import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load the model and tokenizer.
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,
)
model.eval()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Prepare the chat-formatted input.
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

model_inputs = tokenizer(
    [text],
    return_tensors="pt",
    add_special_tokens=False,
).to(model.device)

input_ids = model_inputs["input_ids"]

# 3. Inspect the embedding lookup.
embedding_layer = model.get_input_embeddings()

with torch.inference_mode():
    embeddings = embedding_layer(input_ids)

print("\nToken IDs shape:", input_ids.shape)
print("Embedding table shape:", embedding_layer.weight.shape)
print("Input embeddings shape:", embeddings.shape)

# 4. Inspect the model's dimensions.
config = model.config

print("\nHidden dimension:", config.hidden_size)
print("Query heads:", config.num_attention_heads)
print("Key/value heads:", config.num_key_value_heads)
print("Transformer layers:", config.num_hidden_layers)

# 5. Inspect Q/K/V projections in the first Transformer layer.
first_layer = model.model.layers[0]
attention = first_layer.self_attn

with torch.inference_mode():
    normalized = first_layer.input_layernorm(embeddings)

    queries = attention.q_proj(normalized)
    keys = attention.k_proj(normalized)
    values = attention.v_proj(normalized)

print("\nNormalized input:", normalized.shape)
print("Queries:", queries.shape)
print("Keys:", keys.shape)
print("Values:", values.shape)
