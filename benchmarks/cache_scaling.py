"""Measure dynamic KV scaling or compare dynamic/static allocation on CPU."""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.common import write_json
from benchmarks.warm import describe_runtime


def new_result(max_new_tokens=16, *, compare_static=False):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be an integer between 1 and 128.")
    if compare_static and max_new_tokens > 16:
        raise ValueError("The fixed static-cache comparison supports output caps from 1 to 16.")
    cases = (
        ((1, 8, None), (1, 8, 160), (1, 8, 256)) if compare_static
        else ((1, 1, None), (1, 8, None), (1, 32, None), (2, 8, None), (4, 8, None))
    )
    return {
        "schema_version": 1,
        "kind": "cache_allocation_comparison" if compare_static else "dynamic_cache_scaling",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "runtime": None,
        "max_new_tokens": max_new_tokens,
        "memory_boundary": (
            "Sum of numel * element_size for every layer's K/V tensors after prefill and "
            "final decode; includes dense padding, finished rows and unused static capacity. "
            "Processed positions and backing-tensor capacity are recorded separately. "
            "These are logical tensor bytes, not process RSS. CUDA peaks are absolute PyTorch allocated/reserved bytes, "
            "including weights and temporary tensors: synchronize/reset after releasing "
            "each prefill warm-up, synchronize/read after manual generation, before the "
            "reference. No empty_cache; reservations from warm-ups and earlier cases, "
            "including their reference calls, can persist. GPU peaks are null on CPU."
        ),
        "timing_boundary": (
            "Diagnostic single-run timings, not a performance benchmark: one discarded "
            "prefill warm-up per case; prefill_ms covers the prompt forward only; decode_ms "
            "starts after selecting token 1 and covers subsequent forwards and selections. "
            "CUDA synchronized at timing boundaries. Cache inspection, serialization and "
            "reference generation are outside the timers."
        ),
        "cases": [
            {
                "batch_size": size,
                "prompt_repeats": repeats,
                "cache_capacity": capacity,
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
            for size, repeats, capacity in cases
        ],
    }


def measure(runtime, record):
    """Retain completed measurements if a later case fails or is interrupted."""
    import torch
    from src.batched_generation import generate_batch
    from src.cached_generation import generate_cached
    from src.inference import ContextLimitError

    if runtime.model.device.type not in ("cpu", "cuda") or runtime.model.dtype != torch.float32:
        raise ValueError("This experiment requires CPU or CUDA float32.")
    compare_static = record["kind"] == "cache_allocation_comparison"
    if compare_static and runtime.model.device.type != "cpu":
        raise ValueError("The static-cache comparison is CPU-only.")
    record["runtime"] = describe_runtime(runtime, {"max_tokens": record["max_new_tokens"]})
    del record["runtime"]["batch_size"]
    config = runtime.model.config
    record["cache_configuration"] = {
        "layers": config.num_hidden_layers,
        "query_heads": config.num_attention_heads,
        "kv_heads": config.num_key_value_heads,
        "head_dim": getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
    }
    baseline = None
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
                cache_capacity=case["cache_capacity"],
            )
            case["manual"] = manual
            case["peak_allocated_bytes"] = manual.pop("peak_allocated_bytes")
            case["peak_reserved_bytes"] = manual.pop("peak_reserved_bytes")
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
                "capacity_matches": all(
                    snapshot["allocated_positions"] == (
                        snapshot["cached_positions"] if case["cache_capacity"] is None else case["cache_capacity"]
                    ) and snapshot["cached_positions"] <= snapshot["allocated_positions"]
                    for snapshot in (manual["prefill"], manual["final"])
                ),
            }
            if compare_static:
                if baseline is None:
                    baseline = manual
                case["checks"]["matches_dynamic_baseline"] = all(
                    manual[key] == baseline[key] for key in ("generated_ids", "finish_reasons", "prompt_tokens")
                )
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--compare-static", action="store_true",
                        help="CPU: compare dynamic allocation with static capacities 160 and 256")
    args = parser.parse_args(argv)
    if args.compare_static and args.device != "cpu":
        parser.error("--compare-static requires --device cpu.")
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("Set HF_HUB_OFFLINE=1; use already-cached model artifacts.")
    if Path(args.output).exists():
        parser.error("Choose a fresh output path; previous evidence is never overwritten.")
    try:
        record = new_result(args.max_new_tokens, compare_static=args.compare_static)
    except ValueError as error:
        parser.error(str(error))
    try:
        import torch
        from src.batched_generation import REVISION
        from src.inference import load_runtime

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The requested device requires CUDA.")
        if args.device == "cpu":
            torch.set_num_threads(6)
        runtime = load_runtime(revision=REVISION)
        runtime.model.to(args.device)
        if runtime.model.device.type != args.device:
            raise ValueError("The loaded model is not on the requested device.")
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
