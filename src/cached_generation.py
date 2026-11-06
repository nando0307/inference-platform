"""Manual greedy generation with dynamic/static KV-cache inspection."""

from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StaticCache


def _synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_snapshot(cache, config, positions, capacity):
    """Separate processed positions from logical backing-tensor capacity/bytes."""
    first_layer = cache.layers[0]
    keys = first_layer.keys
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    bytes_per_position = (
        2 * config.num_hidden_layers * keys.shape[0] * config.num_key_value_heads
        * head_dim * keys.element_size()
    )
    return {
        "cache_type": type(cache).__name__,
        "cached_positions": positions,
        "reported_cache_length": int(cache.get_seq_length()),
        "allocated_positions": keys.shape[-2],
        "cached_layers": len(cache.layers),
        "first_key_shape": list(keys.shape),
        "first_value_shape": list(first_layer.values.shape),
        "dtype": str(keys.dtype),
        "bytes_per_element": keys.element_size(),
        "predicted_used_bytes": bytes_per_position * positions,
        "predicted_bytes": bytes_per_position * (positions if capacity is None else capacity),
        "measured_bytes": sum(
            tensor.numel() * tensor.element_size()
            for layer in cache.layers for tensor in (layer.keys, layer.values)
        ),
    }


@torch.inference_mode()
def generate_cached(model, model_inputs, *, max_new_tokens, pad_token_id, cache_capacity=None, fixed_length=False):
    """Inspect cached decoding; fixed_length suppresses EOS for speed workloads only."""
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be an integer between 1 and 512.")
    if pad_token_id is None:
        raise ValueError("A padding token is required for finished batch rows.")
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    if input_ids.ndim != 2 or not all(input_ids.shape) or attention_mask.shape != input_ids.shape:
        raise ValueError("Provide nonempty batched input IDs and a matching attention mask.")
    if (
        not torch.all((attention_mask == 0) | (attention_mask == 1))
        or not torch.all(attention_mask[:, -1] == 1)
        or torch.any(attention_mask[:, :-1] > attention_mask[:, 1:])
    ):
        raise ValueError("Attention masks must describe nonempty, left-padded sequences.")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt width plus output cap exceeds the model context limit.")
    if cache_capacity is not None and (
        type(cache_capacity) is not int
        or not input_ids.shape[1] + max_new_tokens - 1 <= cache_capacity <= model.config.max_position_embeddings
    ):
        raise ValueError("Static capacity must fit all processed positions and stay within the model context limit.")

    eos_token_ids = model.generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos = torch.tensor(eos_token_ids, device=input_ids.device, dtype=input_ids.dtype)
    finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)
    generated = []
    # Masked positions do not advance a sequence's rotary position, as in generate().
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    # Cache writes count padded slots, independently of per-row rotary positions.
    cache_position = torch.arange(input_ids.shape[1], device=input_ids.device)
    past_key_values = StaticCache(config=model.config, max_cache_len=cache_capacity) if cache_capacity is not None else None

    # Warm up the prompt-forward path. Do not reuse this run's cache.
    warmup_outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, cache_position=cache_position,
        past_key_values=past_key_values, use_cache=True,
    )
    del warmup_outputs, past_key_values
    _synchronize(model.device)
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
    past_key_values = StaticCache(config=model.config, max_cache_len=cache_capacity) if cache_capacity is not None else None
    prefill_start = perf_counter()
    # Prefill: process the entire prompt once.
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        position_ids=position_ids, cache_position=cache_position,
        past_key_values=past_key_values, use_cache=True,
    )
    _synchronize(model.device)
    prefill_seconds = perf_counter() - prefill_start
    past_key_values = outputs.past_key_values
    # Track processed slots explicitly: older StaticCache length queries count
    # nonzero keys and can undercount zero-valued padding or legitimate zero keys.
    processed_positions = input_ids.shape[1]
    prefill = _cache_snapshot(past_key_values, model.config, processed_positions, cache_capacity)
    decode_start = None
    decode_seconds = 0.0

    for step in range(max_new_tokens):
        # Select from prefill logits first, then from the previous decode step.
        if fixed_length:
            outputs.logits[:, -1, eos] = -torch.inf
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        next_token = torch.where(finished[:, None], pad_token_id, next_token)
        generated.append(next_token)
        finished |= torch.isin(next_token[:, 0], eos)

        # Selection does not grow the cache. Stop before an unnecessary forward pass.
        if bool(finished.all()) or step + 1 == max_new_tokens:
            break
        # Start after selecting token 1, only if another token is needed.
        if step == 0:
            _synchronize(model.device)
            decode_start = perf_counter()
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
        position_ids = position_ids[:, -1:] + 1
        cache_position = cache_position[-1:] + 1
        # Dense batches keep finished rows; their EOS/padding still occupies cache positions.
        outputs = model(
            input_ids=next_token, attention_mask=attention_mask,
            position_ids=position_ids, cache_position=cache_position,
            past_key_values=past_key_values, use_cache=True,
        )
        past_key_values = outputs.past_key_values
        processed_positions += 1

    if decode_start is not None:
        _synchronize(model.device)
        decode_seconds = perf_counter() - decode_start
    final = _cache_snapshot(past_key_values, model.config, processed_positions, cache_capacity)
    generated_ids = []
    finish_reasons = []
    for row in torch.cat(generated, dim=1).tolist():
        finish = "length"
        for index, token_id in enumerate(row):
            if token_id in eos_token_ids:
                row = row[:index + 1]
                finish = "stop"
                break
        generated_ids.append(row)
        finish_reasons.append(finish)
    result = {
        "prompt_tokens": model_inputs["attention_mask"].sum(dim=1).tolist(),
        "generated_ids": generated_ids,
        "finish_reasons": finish_reasons,
        "prefill": prefill,
        "final": final,
        "prefill_ms": prefill_seconds * 1000,
        "decode_ms": decode_seconds * 1000,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if model.device.type == "cuda":
        # Capture before the caller can run a reference or another workload.
        _synchronize(model.device)
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(model.device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(model.device)
    return result


def main():
    device = "cpu"  # Change to "mps" for the original Apple GPU comparison.
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()
    print("Model device:", model.device)
    print("Model dtype:", model.dtype)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant"},
        {"role": "user", "content": "Reply with exactly one word: Hello"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    max_new_tokens = 100
    result = generate_cached(
        model, model_inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = result["generated_ids"][0]
    prefill = result["prefill"]
    final = result["final"]
    print("Input IDs shape:", model_inputs["input_ids"].shape)
    print("Cache type:", prefill["cache_type"])
    print("Cached sequence length:", prefill["cached_positions"])
    print("First-layer keys shape:", prefill["first_key_shape"])
    print("First-layer values shape:", prefill["first_value_shape"])
    print("Cache dtype:", prefill["dtype"])
    print("Bytes per element:", prefill["bytes_per_element"])
    print("Cached layers:", prefill["cached_layers"])
    print(f"KV storage after prefill: {prefill['measured_bytes'] / 1024**2:.2f} MiB")
    print(f"Prompt forward-pass time: {result['prefill_ms']:.2f} ms")
    print(f"Subsequent generation time: {result['decode_ms']:.2f} ms")
    print(f"Tokens generated after the first: {max(0, len(generated_ids) - 1)}")
    print("Final cache length:", final["cached_positions"])
    stop_reason = "end-of-sequence token" if result["finish_reasons"][0] == "stop" else "maximum new tokens"
    print(f"Stopped: {stop_reason}.")
    print("\nResponse:")
    print(tokenizer.decode(generated_ids, skip_special_tokens=True))
    print(f"KV storage after generation: {final['measured_bytes'] / 1024**2:.2f} MiB")
    comparison_config = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        max_new_tokens=max_new_tokens, use_cache=True,
        eos_token_id=model.generation_config.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        temperature=1.0, top_p=1.0, top_k=50,
    )
    with torch.inference_mode():
        reference_output = model.generate(**model_inputs, generation_config=comparison_config)
    prompt_length = model_inputs["input_ids"].shape[1]
    reference_ids = reference_output[0, prompt_length:].tolist()
    print("\nManual generated tokens:", len(generated_ids))
    print("Hugging Face generated tokens:", len(reference_ids))
    print("Exact token-ID match:", generated_ids == reference_ids)
    assert generated_ids == reference_ids, "Manual and Hugging Face token IDs differ."


if __name__ == "__main__":
    main()
