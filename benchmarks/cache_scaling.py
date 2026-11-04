"""Compare predicted and measured dynamic KV bytes across five CPU workloads."""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmarks.common import write_json
from benchmarks.warm import describe_runtime
from src.batched_generation import REVISION, generate_batch
from src.cached_generation import generate_cached
from src.inference import ContextLimitError, load_runtime


def new_result(max_new_tokens=16):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be an integer between 1 and 128.")
    return {
        "schema_version": 1,
        "kind": "dynamic_cache_scaling",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "runtime": None,
        "max_new_tokens": max_new_tokens,
        "memory_boundary": (
            "Sum of numel * element_size for every layer's K/V tensors after prefill and "
            "final decode; includes dense padding and finished rows; not allocator memory "
            "or process RSS. GPU allocated/reserved peaks are not measured on CPU."
        ),
        "timing_boundary": (
            "Diagnostic single-run timings, not a performance benchmark: one discarded "
            "prefill warm-up per case; prefill_ms covers the prompt forward only; decode_ms "
            "starts after selecting token 1 and covers subsequent forwards and selections. "
            "Cache inspection, serialization and reference generation are outside the timers."
        ),
        "cases": [
            {
                "batch_size": size,
                "prompt_repeats": repeats,
                "prompt": (
                    "An inference server tokenizes prompts and retains attention keys and values. "
                    * repeats + "\nExplain how the cache grows during decoding."
                ),
                "status": "not_run",
                "manual": None,
                "reference": None,
                "checks": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }
            for size, repeats in ((1, 1), (1, 8), (1, 32), (2, 8), (4, 8))
        ],
    }


def measure(runtime, record):
    """Retain completed measurements if a later case fails or is interrupted."""
    if runtime.model.device.type != "cpu" or runtime.model.dtype != torch.float32:
        raise ValueError("This experiment requires CPU float32.")
    record["runtime"] = describe_runtime(runtime, {"max_tokens": record["max_new_tokens"]})
    del record["runtime"]["batch_size"]
    config = runtime.model.config
    record["cache_configuration"] = {
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
    }
    for case in record["cases"]:
        case["status"] = "running"
        try:
            conversations = [[{"role": "user", "content": case["prompt"]}] for _ in range(case["batch_size"])]
            texts = [runtime.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ) for messages in conversations]
            inputs = runtime.tokenizer(
                texts, padding=True, padding_side="left", return_tensors="pt",
                add_special_tokens=False, return_attention_mask=True, truncation=False,
            ).to(runtime.model.device)
            width = inputs["input_ids"].shape[1]
            if width + record["max_new_tokens"] > runtime.context_limit:
                raise ContextLimitError("Prompt width plus output cap exceeds the experiment context limit.")
            manual = generate_cached(
                runtime.model, inputs, max_new_tokens=record["max_new_tokens"],
                pad_token_id=runtime.tokenizer.pad_token_id,
            )
            case["manual"] = manual
            reference = generate_batch(runtime, conversations, max_new_tokens=record["max_new_tokens"])
            case["reference"] = [asdict(result) for result in reference]
            generated_counts = [len(ids) for ids in manual["generated_ids"]]
            case["completion_tokens"] = generated_counts
            case["checks"] = {
                "token_ids_match": manual["generated_ids"] == [result.generated_ids for result in reference],
                "finish_reasons_match": manual["finish_reasons"] == [result.finish_reason for result in reference],
                "prompt_counts_match": manual["prompt_tokens"] == [result.prompt_tokens for result in reference],
                "prefill_bytes_match": manual["prefill"]["measured_bytes"] == manual["prefill"]["predicted_bytes"],
                "final_bytes_match": manual["final"]["measured_bytes"] == manual["final"]["predicted_bytes"],
                "cache_length_matches": (
                    manual["prefill"]["cached_positions"] == width
                    and manual["final"]["cached_positions"] == width + max(generated_counts) - 1
                ),
            }
            case["status"] = "success" if all(case["checks"].values()) else "mismatch"
        except (Exception, KeyboardInterrupt) as error:
            case["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
            case["exception_type"] = type(error).__name__
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            return record
    record["status"] = "completed" if all(case["status"] == "success" for case in record["cases"]) else "failed"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("Set HF_HUB_OFFLINE=1; use already-cached model artifacts.")
    if Path(args.output).exists():
        parser.error("Choose a fresh output path; previous evidence is never overwritten.")
    try:
        record = new_result(args.max_new_tokens)
    except ValueError as error:
        parser.error(str(error))
    try:
        torch.set_num_threads(6)
        runtime = load_runtime(revision=REVISION)
        runtime.benchmark_model_revision = REVISION
        measure(runtime, record)
    except KeyboardInterrupt:
        record.update(status="interrupted", exception_type="KeyboardInterrupt")
    except Exception as error:
        record["exception_type"] = type(error).__name__
    write_json(args.output, record)
    print(json.dumps(record, indent=2))
    print(f"Saved {args.output}")
    return 0 if record["status"] == "completed" else 130 if record["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
