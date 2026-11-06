"""Compare CPU FP32, FP16 and QNNPACK dynamic INT8 speed and task success offline."""

import argparse
import gc
import json
import os
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.common import write_json
from benchmarks.warm import describe_runtime
from benchmarks.workloads import LONG_PROMPT, SHORT_PROMPT


# Frozen before observing trained-model answers. These are task checks, not a general benchmark.
QUALITY_CASES = (
    {"name": "addition", "category": "arithmetic", "prompt": "What is 17 + 28? Reply with only the integer.", "expected": ["45"]},
    {"name": "multiplication", "category": "arithmetic", "prompt": "What is 9 times 7? Reply with only the integer.", "expected": ["63"]},
    {"name": "division", "category": "arithmetic", "prompt": "What is 144 divided by 12? Reply with only the integer.", "expected": ["12"]},
    {"name": "capital", "category": "facts", "prompt": "What is the capital of France? Reply with only the city name.", "expected": ["paris"]},
    {"name": "water", "category": "facts", "prompt": "What is the chemical formula of water? Reply with only the formula.", "expected": ["h2o", "h₂o"]},
    {"name": "planet", "category": "facts", "prompt": "Which planet is known as the Red Planet? Reply with only the planet name.", "expected": ["mars"]},
    {"name": "json", "category": "instructions", "prompt": 'Return only a JSON object with exactly two keys: "color" with the string value "blue", and "count" with the integer value 2. No markdown or explanation.', "expected": {"color": "blue", "count": 2}},
    {"name": "reverse", "category": "instructions", "prompt": "Return these words in reverse order, separated by single spaces, with no extra text: red green blue", "expected": ["blue green red"]},
    {"name": "lowercase", "category": "instructions", "prompt": "Convert the following text to lowercase. Reply with only the converted text: MIXED Case", "expected": ["mixed case"]},
)


def score_quality(case, text, finish_reason):
    """Require a complete answer; ignore outer whitespace and fact-answer casing only."""
    if finish_reason != "stop":
        return False
    text = text.strip()
    if case["name"] == "json":
        try:
            value = json.loads(text)
        except ValueError:
            return False
        return isinstance(value, dict) and value == case["expected"] and type(value["count"]) is int
    if case["category"] == "facts":
        text = text.casefold()
    return text in case["expected"]


def measure(runtime, record):
    import torch
    from src.cached_generation import generate_cached
    from src.inference import ContextLimitError, generate

    if runtime.model.device.type != "cpu":
        raise ValueError("This precision comparison is CPU-only.")
    record["runtime"] = describe_runtime(runtime, {"max_tokens": 64})
    quantized = [(name, module) for name, module in runtime.model.named_modules()
                 if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)]
    record["precision"] = {
        "floating_parameter_dtypes": sorted({str(p.dtype) for p in runtime.model.parameters()}),
        "quantized_linear_names": [name for name, module in quantized],
        "quantized_linear_count": len(quantized),
        "sampled_quantized_weight_dtype": str(quantized[0][1].weight().dtype) if quantized else None,
        "quantization_engine": torch.backends.quantized.engine if quantized else None,
        "activation_policy": (
            "Dynamic INT8 inside quantized linears; FP32 inputs/outputs, embeddings and norms."
            if quantized else "Floating-point execution; individual operations may accumulate at higher precision."
        ),
        "dtype_field_scope": "runtime.dtype describes floating parameters, not packed quantized weights.",
    }
    for name, prompt, expected_tokens in (("short", SHORT_PROMPT, 59), ("long", LONG_PROMPT, 990)):
        text = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
        )
        inputs = runtime.tokenizer([text], return_tensors="pt", add_special_tokens=False,
                                   return_attention_mask=True, truncation=False)
        width = inputs["input_ids"].shape[1]
        if width != expected_tokens:
            raise ValueError("The pinned speed workload token count changed.")
        if width + 32 > runtime.context_limit:
            raise ContextLimitError("Speed workload exceeds the runtime context limit.")
        # One complete discarded generation also warms the cached-decode kernels.
        generate_cached(runtime.model, inputs, max_new_tokens=32,
                        pad_token_id=runtime.tokenizer.pad_token_id, fixed_length=True)
        result = generate_cached(runtime.model, inputs, max_new_tokens=32,
                                 pad_token_id=runtime.tokenizer.pad_token_id, fixed_length=True)
        record["speed"].append({
            "name": name, "prompt": prompt, "result": result,
            "decode_ms_per_token": result["decode_ms"] / 31,
            "decode_tokens_per_second": 31_000 / result["decode_ms"],
        })
        if len(result["generated_ids"][0]) != 32 or result["finish_reasons"] != ["length"]:
            raise RuntimeError("Fixed-length generation did not produce exactly 32 tokens.")
        print(f'{record["name"]} {name}: prefill={result["prefill_ms"]:.2f} ms, '
              f'decode={result["decode_ms"] / 31:.2f} ms/token', flush=True)
    for case in QUALITY_CASES:
        result = generate(runtime, [{"role": "user", "content": case["prompt"]}],
                          max_new_tokens=64, temperature=0, top_p=1)
        record["quality"].append({
            **case, **asdict(result), "completion_tokens": len(result.generated_ids),
            "truncated": result.finish_reason == "length",
            "passed": score_quality(case, result.text, result.finish_reason),
        })
    record["scores"] = {
        category: {"passed": sum(row["passed"] for row in record["quality"] if row["category"] == category),
                   "total": sum(row["category"] == category for row in record["quality"])}
        for category in ("arithmetic", "facts", "instructions")
    }
    record["status"] = "completed"
    print(f'{record["name"]} quality: {json.dumps(record["scores"])}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("Set HF_HUB_OFFLINE=1; use already-cached model artifacts.")
    if Path(args.output).exists():
        parser.error("Choose a fresh output path; previous evidence is never overwritten.")
    record = {
        "schema_version": 1, "kind": "cpu_precision_comparison",
        "recorded_at": datetime.now(timezone.utc).isoformat(), "status": "failed",
        "speed_protocol": {
            "max_new_tokens": 32, "fixed_length": True, "full_warmups": 1, "measured_samples": 1,
            "boundary": (
                "Manual greedy cached loop, batch 1, EOS suppressed. Each call also discards a prefill warm-up. "
                "prefill_ms covers prompt forward only; decode_ms covers 31 subsequent forwards/selections "
                "after token 1. Excludes loading, conversion, tokenization, cache inspection and serialization. "
                "Single-run observations, not stable speedups or end-to-end serving latency."
            ),
        },
        "quality_protocol": {
            "max_new_tokens": 64, "stopping": "natural EOS or output cap", "cases": QUALITY_CASES,
            "scoring": (
                "Task success, not token agreement: outer whitespace ignored; fact-answer casing ignored; "
                "JSON parsed with exact keys/values and integer count; other answers exact. "
                "Truncated outputs fail. Raw answers retained. Nine simple tasks are not a general quality benchmark."
            ),
        },
        "quantization": {
            "method": "torch.ao.quantization.quantize_dynamic", "modules": "all torch.nn.Linear, including lm_head",
            "weight_dtype": "torch.qint8", "engine": "qnnpack",
            "caveat": "QNNPACK warns that it ignores reduce_range=True; warning is not suppressed. Not weight-only quantization.",
        },
        "formats": [{"name": name, "status": "not_run", "runtime": None, "speed": [], "quality": []}
                    for name in ("fp32", "fp16", "dynamic-int8")],
    }
    try:
        import torch
        from src.batched_generation import REVISION
        from src.inference import load_runtime

        torch.set_num_threads(6)
        for format_record in record["formats"]:
            format_record["status"] = "running"
            print(f'Loading {format_record["name"]} on CPU...', flush=True)
            runtime = load_runtime(revision=REVISION)
            if runtime.model.device.type != "cpu" or runtime.model.dtype != torch.float32:
                raise ValueError("Load the CPU FP32 checkpoint before experimental conversion.")
            runtime.benchmark_model_revision = REVISION
            if format_record["name"] == "fp16":
                runtime.model.half()
            elif format_record["name"] == "dynamic-int8":
                if "qnnpack" not in torch.backends.quantized.supported_engines:
                    raise RuntimeError("This comparison requires the QNNPACK engine.")
                torch.backends.quantized.engine = "qnnpack"
                torch.ao.quantization.quantize_dynamic(
                    runtime.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True,
                )
            measure(runtime, format_record)
            del runtime
            gc.collect()
        record["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["exception_type"] = type(error).__name__
        for format_record in record["formats"]:
            if format_record["status"] == "running":
                format_record["status"] = record["status"]
        traceback.print_exc()
    write_json(args.output, record)
    print(f"Saved {args.output}")
    return 0 if record["status"] == "completed" else 130 if record["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
