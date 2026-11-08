"""Two explicitly selected, approval-gated routing phases on one private T4."""

import json
from pathlib import Path

import modal

from benchmarks.common import write_json
from modal_continuous_batching import ENGINE_IMAGE, ENGINE_VERSION

app = modal.App("inference-qwen-model-routing")


def download_models():
    """CPU image-build downloads only; the GPU functions run strictly offline."""
    from huggingface_hub import snapshot_download
    from src.vllm_engine import MODEL_SPECS

    for model, spec in MODEL_SPECS.items():
        snapshot_download(model, revision=spec["revision"],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


# Start from the same digest, not the inherited image's copy=False runtime mounts.
image = (
    modal.Image.from_registry(ENGINE_IMAGE)
    .entrypoint([])
    .run_commands("ln -s /usr/bin/python3 /usr/local/bin/python")
    .env({"HF_HOME": "/opt/huggingface", "TOKENIZERS_PARALLELISM": "false", "VLLM_NO_USAGE_STATS": "1"})
    .workdir("/root")
)
for relative in (
    "src/vllm_engine.py", "benchmarks/common.py", "benchmarks/workloads.py",
    "modal_inference.py", "modal_continuous_batching.py",
):
    image = image.add_local_file(Path(__file__).parent / relative, f"/root/{relative}", copy=True)
image = image.run_function(download_models, timeout=900).env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
for relative in (
    "src/api.py", "src/inference.py", "src/streaming.py", "src/model_router.py",
    "benchmarks/client.py", "benchmarks/continuous_batching.py", "benchmarks/precision.py",
    "benchmarks/warm.py", "benchmarks/model_routing.py", "modal_model_routing.py",
):
    image = image.add_local_file(Path(__file__).parent / relative, f"/root/{relative}")


def _run(mode):
    import asyncio
    from importlib.metadata import version

    from benchmarks.model_routing import LIMITS, _fail, new_result, run_experiment

    record = new_result(mode)
    record["runtime"].update(
        image=ENGINE_IMAGE, engine_version=ENGINE_VERSION,
        resource_limits={"gpu": "T4", "cpu": 4, "memory_mib": 8192, "max_containers": 1,
                         "min_containers": 0, "scaledown_window_seconds": 2,
                         "function_timeout_seconds": LIMITS[mode]["function_seconds"], "retries": 0},
    )
    try:
        if version("vllm") != ENGINE_VERSION:
            raise RuntimeError("The image does not contain the pinned vLLM version.")
        asyncio.run(run_experiment(record))
    except (Exception, KeyboardInterrupt) as error:
        _fail(record, error)
    return record


@app.function(image=image, gpu="T4", cpu=4, memory=8192,
              max_containers=1, min_containers=0, scaledown_window=2, timeout=600, retries=0)
def smoke():
    return _run("smoke")


@app.function(image=image, gpu="T4", cpu=4, memory=8192,
              max_containers=1, min_containers=0, scaledown_window=2, timeout=900, retries=0)
def lifecycle():
    return _run("lifecycle")


@app.local_entrypoint()
def main(mode: str, output: str):
    from benchmarks.model_routing import _fail, new_result

    record = new_result(mode)
    if Path(output).exists():
        raise FileExistsError("Choose a fresh output path; previous evidence is never overwritten.")
    record["remote_results_received"] = False
    try:
        record.update((smoke if mode == "smoke" else lifecycle).remote())
        record["remote_results_received"] = True
    except (Exception, KeyboardInterrupt) as error:
        _fail(record, error)
    finally:
        write_json(output, record)
    print(json.dumps({"status": record["status"], "checks": record["checks"], "output": output}, indent=2))
    if record["status"] != "completed" or record["checks"].get("all_passed") is not True:
        raise SystemExit(130 if record["status"] == "interrupted" else 1)
