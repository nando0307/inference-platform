"""Measure and save one bounded, warm Qwen experiment on a T4."""

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

app = modal.App("inference-qwen-check")


def download_model():
    """Cache only inference artifacts in the image, without allocating a GPU."""
    from huggingface_hub import snapshot_download
    from src.inference import MODEL_ID

    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0+cu124",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==4.57.6",
        "numpy==2.2.6",
        "huggingface-hub==0.36.0",
        "tokenizers==0.22.1",
        "safetensors==0.6.2",
    )
    .env({"HF_HOME": "/opt/huggingface", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(
        Path(__file__).parent / "src" / "inference.py",
        "/root/src/inference.py",
        copy=True,
    )
    .run_function(download_model, timeout=600)
    .env({"HF_HUB_OFFLINE": "1"})
)

# Add benchmark code after the cached download layer, without uploading result directories.
for module in ("common", "model", "timing", "gpu"):
    image = image.add_local_file(
        Path(__file__).parent / "benchmarks" / f"{module}.py",
        f"/root/benchmarks/{module}.py",
    )


@app.function(
    image=image,
    gpu="T4",
    memory=4096,
    max_containers=1,
    min_containers=0,
    timeout=180,
    retries=0,
)
def run_inference(manifest):
    import hashlib
    import platform
    from importlib.metadata import version

    import torch
    from benchmarks.gpu import run_samples, select_case
    from src.inference import MODEL_ID, load_runtime

    assert torch.cuda.is_available(), "This experiment requires CUDA."
    runtime = load_runtime(revision=REVISION)
    runtime.model.to("cuda")
    assert runtime.model.device.type == "cuda"
    assert runtime.model.dtype == torch.float32
    runtime.benchmark_model_revision = REVISION
    case = select_case(manifest, runtime)

    metadata = {
        "model": MODEL_ID,
        "revision": REVISION,
        "chat_template_sha256": hashlib.sha256(
            runtime.tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "gpu": torch.cuda.get_device_name(0),
        "device": str(runtime.model.device),
        "dtype": str(runtime.model.dtype),
        "python_version": platform.python_version(),
        "torch_cuda_version": torch.version.cuda,
        "package_versions": {
            name: version(name)
            for name in (
                "torch", "transformers", "numpy", "huggingface-hub",
                "tokenizers", "safetensors",
            )
        },
        "generation_settings": {"max_new_tokens": case["max_tokens"], "temperature": 0, "top_p": 1},
        "context_limit": runtime.context_limit,
        "batch_size": 1,
        "concurrency": 1,
        "eos_counted": True,
        "stopping": "natural EOS or output cap",
    }
    return {"runtime": metadata, "samples": run_samples(runtime, case)}


@app.local_entrypoint()
def main(output: str, workload: str = "benchmark-results/phase3/workload.json"):
    from benchmarks.common import write_json
    from benchmarks.gpu import CASE, REPEATS, TIMING_BOUNDARY, WARMUPS, summarize

    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError("Choose a fresh output path; previous evidence is never overwritten.")
    with Path(workload).open(encoding="utf-8") as source:
        manifest = json.load(source)
    payload = {
        "schema_version": 1,
        "kind": "gpu_warm_inference",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workload": manifest,
        "case": CASE,
        "warmups": WARMUPS,
        "repeats": REPEATS,
        "timing_boundary": TIMING_BOUNDARY,
        "status": "failed",
        "remote_results_received": False,
        "runtime": None,
        "samples": [],
    }
    try:
        payload.update(run_inference.remote(manifest))
        payload["remote_results_received"] = True
    except KeyboardInterrupt:
        payload.update(status="interrupted", exception_type="KeyboardInterrupt")
    except Exception as error:
        payload["exception_type"] = type(error).__name__
    payload["summary"] = (
        summarize(payload["samples"]) if payload["remote_results_received"] else None
    )
    if payload["remote_results_received"]:
        payload["status"] = "completed" if payload["summary"]["complete"] else "failed"
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))
    print(f"Saved {output_path}")
    if payload["status"] != "completed":
        raise SystemExit(130 if payload["status"] == "interrupted" else 1)
