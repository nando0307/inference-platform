"""Bounded warm GPU measurements; importing this module does not load ML libraries."""

from time import perf_counter

from benchmarks.common import distribution

CASE = "short-input-long-output"
WARMUPS = 2
REPEATS = 5
TIMING_BOUNDARY = (
    "src.inference.generate including formatting, tokenization, input transfer, "
    "prefill, decoding, output transfer and text decoding; CUDA synchronized "
    "before starting and before stopping the clock; excludes model loading, "
    "warm-ups and Modal invocation/network overhead"
)


def select_case(manifest, runtime):
    from benchmarks.model import validate_workload

    if not isinstance(manifest.get("chat_template_sha256"), str):
        raise ValueError("The workload must record its chat-template hash.")
    case, = validate_workload(manifest, runtime, [CASE])
    if case["prompt_tokens"] != 59 or case["max_tokens"] != 128:
        raise ValueError("This experiment requires the fixed 59/128 workload.")
    return case


def run_samples(runtime, case):
    import torch
    from src.inference import generate

    samples = []
    for phase, count in (("warmup", WARMUPS), ("measured", REPEATS)):
        for iteration in range(count):
            row = {
                "case": case["name"], "phase": phase, "iteration": iteration,
                "status": "error", "warm_inference_ms": None,
                "prompt_tokens": None, "completion_tokens": None,
                "finish_reason": None,
            }
            try:
                torch.cuda.synchronize()
                if phase == "measured":
                    started = perf_counter()
                result = generate(
                    runtime, case["messages"], max_new_tokens=case["max_tokens"],
                    temperature=0, top_p=1,
                )
                torch.cuda.synchronize()
                elapsed_ms = (perf_counter() - started) * 1000 if phase == "measured" else None
                if result.prompt_tokens != case["prompt_tokens"]:
                    raise ValueError("Generated prompt count differs from the fixed workload.")
                row.update(
                    status="success", warm_inference_ms=elapsed_ms,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=len(result.generated_ids),
                    finish_reason=result.finish_reason,
                )
            except Exception as error:
                row["exception_type"] = type(error).__name__
                samples.append(row)
                # A failed CUDA operation may leave the context unusable; do not retry.
                return samples
            samples.append(row)
    return samples


def summarize(samples):
    warmups = [row for row in samples if row["phase"] == "warmup"]
    measured = [row for row in samples if row["phase"] == "measured"]
    successful = [row for row in measured if row["status"] == "success"]
    durations = [row["warm_inference_ms"] for row in successful]
    elapsed_ms = sum(durations)
    completed_warmups = sum(row["status"] == "success" for row in warmups)
    return {
        "complete": completed_warmups == WARMUPS and len(successful) == REPEATS
        and len(samples) == WARMUPS + REPEATS,
        "warmup_attempts": len(warmups),
        "warmup_errors": len(warmups) - completed_warmups,
        "measured_attempts": len(measured),
        "successful_measurements": len(successful),
        "measured_errors": len(measured) - len(successful),
        "median_warm_inference_ms": distribution(durations)["p50"],
        "successful_measured_ms": elapsed_ms if successful else None,
        "completion_tokens_per_second": (
            sum(row["completion_tokens"] for row in successful) * 1000 / elapsed_ms
            if elapsed_ms > 0 else None
        ),
        "throughput_boundary": (
            "successful measured output tokens / sum of successful warm inference seconds; "
            "excludes warm-ups, failures and time between calls; not decode-only throughput"
        ),
    }
