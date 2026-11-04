"""Approval-gated T4 cache-scaling experiment; importing this file launches no job."""

import json
from pathlib import Path

import modal

from benchmarks.cache_scaling import new_result
from benchmarks.common import write_json
from modal_inference import REVISION, image

app = modal.App("inference-qwen-cache-scaling")
# Reuse the pinned image/download layer; never mount local result directories.
for relative in (
    "benchmarks/cache_scaling.py", "benchmarks/workloads.py",
    "src/batched_generation.py", "src/cached_generation.py",
):
    image = image.add_local_file(Path(__file__).parent / relative, f"/root/{relative}")


@app.function(
    image=image, gpu="T4", memory=4096,
    max_containers=1, min_containers=0, timeout=300, retries=0,
)
def run_experiment(max_new_tokens: int = 16):
    import torch
    from benchmarks.cache_scaling import measure
    from src.inference import load_runtime

    record = new_result(max_new_tokens)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA.")
    runtime = load_runtime(revision=REVISION)
    runtime.model.to("cuda")
    if runtime.model.device.type != "cuda" or runtime.model.dtype != torch.float32:
        raise ValueError("This experiment requires CUDA float32.")
    runtime.benchmark_model_revision = REVISION
    return measure(runtime, record)


@app.local_entrypoint()
def main(output: str, max_new_tokens: int = 16):
    if Path(output).exists():
        raise FileExistsError("Choose a fresh output path; previous evidence is never overwritten.")
    # Reject invalid bounds before submitting remote work.
    record = new_result(max_new_tokens)
    record["remote_results_received"] = False
    try:
        record.update(run_experiment.remote(max_new_tokens))
        record["remote_results_received"] = True
    except KeyboardInterrupt:
        record.update(status="interrupted", exception_type="KeyboardInterrupt")
    except Exception as error:
        record["exception_type"] = type(error).__name__
    write_json(output, record)
    print(json.dumps(record, indent=2))
    print(f"Saved {output}")
    if record["status"] != "completed":
        raise SystemExit(130 if record["status"] == "interrupted" else 1)
