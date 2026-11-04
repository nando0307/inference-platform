"""Measure the same four prompts sequentially and in static batches of two/four."""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from benchmarks.common import distribution, write_json
from benchmarks.warm import describe_runtime
from benchmarks.workloads import SHORT_PROMPT

PROMPTS = (
    "Reply with exactly one word: Hello",
    SHORT_PROMPT,
    "What is 2 + 2? Reply with just the number.",
    "Explain in two sentences why attention uses a KV cache during autoregressive decoding.",
)


def new_result(max_new_tokens=32, warmups=1, repeats=3):
    for name, value, maximum in (
        ("max_new_tokens", max_new_tokens, 128), ("warmups", warmups, 2), ("repeats", repeats, 5),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
    return {
        "schema_version": 1,
        "kind": "static_batch_benchmark",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "runtime": None,
        "conversations": [[{"role": "user", "content": prompt}] for prompt in PROMPTS],
        "max_new_tokens": max_new_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "configuration_order": [1, 2, 4],
        "timing_boundary": (
            "workload_ms: wall time for all four prompts, including call timing, dispatch and "
            "loop bookkeeping; call_ms: whole generate/generate_batch including formatting, "
            "tokenization, transfers, prefill, decode and text decoding; CUDA synchronized "
            "at boundaries; excludes loading, warm-ups, result serialization and comparisons; "
            "not HTTP latency, TTFT, TPOT or batch duration divided by batch size"
        ),
        "memory_boundary": (
            "CUDA absolute peak allocated/reserved bytes from PyTorch's allocator during each "
            "measured four-prompt workload, counters reset after warm-up and before timing; "
            "includes weights, excludes non-PyTorch allocations; cached reservations from "
            "earlier configurations remain; no empty_cache; CPU memory is not measured"
        ),
        "reference_results": None,
        "configurations": [{"batch_size": size, "samples": [], "summary": None} for size in (1, 2, 4)],
    }


def summarize(samples, warmups, repeats):
    measured = [sample for sample in samples if sample["phase"] == "measured"]
    successful = [sample for sample in measured if sample["status"] == "success"]
    warmed = sum(sample["phase"] == "warmup" and sample["status"] == "success" for sample in samples)
    total_ms = sum(sample["workload_ms"] for sample in successful)
    tokens = sum(sample["completion_tokens"] for sample in successful)
    calls = [call for sample in successful for call in sample["calls"]]
    return {
        "complete": warmed == warmups and len(successful) == repeats,
        "successful_warmups": warmed,
        "warmup_failures": sum(sample["phase"] == "warmup" and sample["status"] != "success" for sample in samples),
        "successful_measured_workloads": len(successful),
        "failed_measured_workloads": len(measured) - len(successful),
        "median_workload_ms": distribution([sample["workload_ms"] for sample in successful])["p50"],
        "median_call_ms": distribution([call["call_ms"] for call in calls])["p50"],
        "successful_measured_ms": total_ms if successful else None,
        "completion_tokens": tokens if successful else None,
        "output_tokens_per_second": tokens * 1000 / total_ms if total_ms > 0 else None,
        "output_mismatches": sum(not row["matches_reference"] for call in calls for row in call["results"]),
        "peak_allocated_bytes": max((sample["peak_allocated_bytes"] for sample in successful
                                     if sample["peak_allocated_bytes"] is not None), default=None),
        "peak_reserved_bytes": max((sample["peak_reserved_bytes"] for sample in successful
                                    if sample["peak_reserved_bytes"] is not None), default=None),
        "throughput_boundary": "actual output tokens / summed complete measured workload seconds; excludes failed workloads",
    }


def measure(runtime, record):
    """Fill record in place so a local interruption retains completed calls."""
    import torch
    from src.batched_generation import generate_batch
    from src.inference import generate

    device = runtime.model.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only synchronous CPU and synchronized CUDA measurements are supported.")
    if len(record["conversations"]) != 4:
        raise ValueError("This experiment requires exactly four conversations.")
    record["runtime"] = describe_runtime(runtime, {"max_tokens": record["max_new_tokens"]})
    del record["runtime"]["batch_size"]
    failed = False
    for configuration in record["configurations"]:
        size = configuration["batch_size"]
        groups = [(list(range(start, start + size)), record["conversations"][start:start + size])
                  for start in range(0, 4, size)]
        for phase, count in (("warmup", record["warmups"]), ("measured", record["repeats"])):
            for iteration in range(count):
                sample = {
                    "phase": phase, "iteration": iteration, "status": "error",
                    "workload_ms": None, "completion_tokens": None,
                    "peak_allocated_bytes": None, "peak_reserved_bytes": None,
                    "calls": [{"input_indices": indices, "status": "not_run", "call_ms": None, "results": []}
                              for indices, _ in groups],
                }
                configuration["samples"].append(sample)
                completed = []
                active = None
                try:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        if phase == "measured":
                            torch.cuda.reset_peak_memory_stats(device)
                    if phase == "measured":
                        workload_started = perf_counter()
                    for call, (_, conversations) in zip(sample["calls"], groups):
                        active = call
                        call["status"] = "running"
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        if phase == "measured":
                            call_started = perf_counter()
                        if size == 1:
                            results = [generate(runtime, conversations[0], max_new_tokens=record["max_new_tokens"],
                                                temperature=0, top_p=1)]
                        else:
                            results = generate_batch(runtime, conversations, max_new_tokens=record["max_new_tokens"])
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        if phase == "measured":
                            call["call_ms"] = (perf_counter() - call_started) * 1000
                        if len(results) != size:
                            raise ValueError("Generation returned the wrong number of responses.")
                        completed.append(results)
                        call["status"] = "success"
                        active = None
                    if phase == "measured":
                        sample["workload_ms"] = (perf_counter() - workload_started) * 1000
                        if device.type == "cuda":
                            sample["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                            sample["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
                    sample["status"] = "success"
                except (Exception, KeyboardInterrupt) as error:
                    sample["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                    sample["exception_type"] = type(error).__name__
                    sample["workload_ms"] = None
                    if active is not None:
                        active.update(status=sample["status"], call_ms=None, exception_type=type(error).__name__)
                    record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                    failed = True
                # Serialization and output comparisons are outside both clocks.
                for call, results in zip(sample["calls"], completed):
                    for index, result in zip(call["input_indices"], results):
                        reference = record["reference_results"]
                        call["results"].append({
                            **asdict(result), "input_index": index,
                            "completion_tokens": len(result.generated_ids),
                            "matches_reference": (
                                reference[index]["generated_ids"] == result.generated_ids
                                and reference[index]["prompt_tokens"] == result.prompt_tokens
                                and reference[index]["finish_reason"] == result.finish_reason
                            ) if reference is not None else None,
                        })
                rows = [row for call in sample["calls"] for row in call["results"]]
                if sample["status"] == "success":
                    sample["completion_tokens"] = sum(row["completion_tokens"] for row in rows)
                    if record["reference_results"] is None:
                        record["reference_results"] = rows
                if failed:
                    break
            if failed:
                break
        if failed:
            break
    for configuration in record["configurations"]:
        configuration["summary"] = summarize(configuration["samples"], record["warmups"], record["repeats"])
    if not failed:
        record["status"] = "completed"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("Set HF_HUB_OFFLINE=1; use already-cached model artifacts.")
    if Path(args.output).exists():
        parser.error("Choose a fresh output path; previous evidence is never overwritten.")
    try:
        record = new_result(args.max_new_tokens, args.warmups, args.repeats)
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
        if runtime.model.device.type != args.device or runtime.model.dtype != torch.float32:
            raise ValueError("The benchmark requires the requested device and float32.")
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
