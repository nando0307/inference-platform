"""Shared warm CPU/CUDA measurements and an offline CPU control command."""

from time import perf_counter

from benchmarks.common import distribution, environment, write_json

CASE = "short-input-long-output"
WARMUPS = 2
REPEATS = 5
TIMING_BOUNDARY = (
    "src.inference.generate including formatting, tokenization, input transfer, "
    "prefill, decoding, output transfer and text decoding; CUDA runs synchronize "
    "before starting and before stopping the clock; CPU execution is synchronous; "
    "excludes model loading, warm-ups and invocation/network overhead"
)


def select_case(manifest, runtime):
    from benchmarks.model import validate_workload

    if not isinstance(manifest.get("chat_template_sha256"), str):
        raise ValueError("The workload must record its chat-template hash.")
    case, = validate_workload(manifest, runtime, [CASE])
    if case["prompt_tokens"] != 59 or case["max_tokens"] != 128:
        raise ValueError("This experiment requires the fixed 59/128 workload.")
    return case


def describe_runtime(runtime, case):
    import hashlib
    from importlib.metadata import version

    import torch
    from src.inference import MODEL_ID

    host = environment()
    return {
        "model": MODEL_ID,
        "revision": runtime.benchmark_model_revision,
        "chat_template_sha256": hashlib.sha256(
            runtime.tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "gpu": (
            torch.cuda.get_device_name(runtime.model.device)
            if runtime.model.device.type == "cuda" else None
        ),
        "device": str(runtime.model.device),
        "dtype": str(runtime.model.dtype),
        "python_version": host["python"],
        "torch_cuda_version": torch.version.cuda,
        "package_versions": {
            name: version(name)
            for name in (
                "torch", "transformers", "numpy", "huggingface-hub",
                "tokenizers", "safetensors",
            )
        },
        "host": {key: host[key] for key in ("platform", "architecture", "cpu", "logical_cpus")},
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "generation_settings": {"max_new_tokens": case["max_tokens"], "temperature": 0, "top_p": 1},
        "context_limit": runtime.context_limit,
        "batch_size": 1,
        "concurrency": 1,
        "eos_counted": True,
        "stopping": "natural EOS or output cap",
    }


def run_samples(runtime, case):
    import torch
    from src.inference import generate

    device = runtime.model.device.type
    if device not in ("cpu", "cuda"):
        raise ValueError("Warm measurements support only synchronous CPU or synchronized CUDA.")

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
                if device == "cuda":
                    torch.cuda.synchronize(runtime.model.device)
                if phase == "measured":
                    started = perf_counter()
                result = generate(
                    runtime, case["messages"], max_new_tokens=case["max_tokens"],
                    temperature=0, top_p=1,
                )
                if device == "cuda":
                    torch.cuda.synchronize(runtime.model.device)
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


def main(argv=None):
    import argparse
    import json
    import os
    from datetime import datetime, timezone
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run the offline CPU control for the warm Modal workload.")
    parser.add_argument("--workload", default="benchmark-results/phase3/workload.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("Set HF_HUB_OFFLINE=1; this control must use already-cached artifacts.")
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a fresh output path; previous evidence is never overwritten.")
    with Path(args.workload).open(encoding="utf-8") as source:
        manifest = json.load(source)
    payload = {
        "schema_version": 1,
        "kind": "cpu_warm_inference",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workload": manifest,
        "case": CASE,
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "timing_boundary": TIMING_BOUNDARY,
        "status": "failed",
        "samples_available": False,
        "runtime": None,
        "samples": [],
    }
    try:
        import torch
        from src.inference import load_runtime

        torch.set_num_threads(6)
        runtime = load_runtime(revision=manifest["model_revision"])
        if runtime.model.device.type != "cpu" or runtime.model.dtype != torch.float32:
            raise ValueError("The local control requires CPU float32.")
        runtime.benchmark_model_revision = manifest["model_revision"]
        case = select_case(manifest, runtime)
        payload["runtime"] = describe_runtime(runtime, case)
        payload["samples"] = run_samples(runtime, case)
        payload["samples_available"] = True
    except KeyboardInterrupt:
        payload.update(status="interrupted", exception_type="KeyboardInterrupt")
    except Exception as error:
        payload["exception_type"] = type(error).__name__
    payload["summary"] = summarize(payload["samples"]) if payload["samples_available"] else None
    if payload["samples_available"]:
        payload["status"] = "completed" if payload["summary"]["complete"] else "failed"
    write_json(output, payload)
    print(json.dumps(payload, indent=2))
    print(f"Saved {output}")
    return 0 if payload["status"] == "completed" else 130 if payload["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
