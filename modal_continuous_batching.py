"""Run one approval-gated, private vLLM continuous-batching experiment on a T4."""

import json
from pathlib import Path

import modal

from benchmarks.common import write_json
from benchmarks.workloads import MODEL_ID
from modal_inference import REVISION


ENGINE_VERSION = "0.33.0"
# Pin the official Linux/amd64 image, including its compatible Torch/CUDA stack.
ENGINE_IMAGE = "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
app = modal.App("inference-qwen-continuous-batching")


def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, revision=REVISION,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])


image = (
    modal.Image.from_registry(ENGINE_IMAGE)
    .entrypoint([])
    # The official image has python3 only; Modal requires python on PATH.
    .run_commands("ln -s /usr/bin/python3 /usr/local/bin/python")
    .env({"HF_HOME": "/opt/huggingface", "TOKENIZERS_PARALLELISM": "false", "VLLM_NO_USAGE_STATS": "1"})
    .add_local_file(Path(__file__).parent / "benchmarks/common.py", "/root/benchmarks/common.py", copy=True)
    .add_local_file(Path(__file__).parent / "benchmarks/workloads.py", "/root/benchmarks/workloads.py", copy=True)
    # Avoid the upstream /vllm-workspace/benchmarks package shadowing ours.
    .workdir("/root")
    .add_local_file(Path(__file__).parent / "modal_inference.py", "/root/modal_inference.py", copy=True)
    .run_function(download_model, timeout=600)
    .env({"HF_HUB_OFFLINE": "1"})
)
for relative in ("benchmarks/client.py", "benchmarks/continuous_batching.py"):
    image = image.add_local_file(Path(__file__).parent / relative, f"/root/{relative}")


@app.function(
    image=image, gpu="T4", cpu=4, memory=8192,
    max_containers=1, min_containers=0, scaledown_window=2, timeout=600, retries=0,
)
def run_benchmark():
    import asyncio
    import hashlib
    import os
    import signal
    import subprocess
    import traceback
    from importlib.metadata import version
    from time import perf_counter

    from benchmarks.common import environment
    from benchmarks.continuous_batching import measure, new_result

    record = new_result()
    record["status"] = "running"
    process = None
    try:
        import httpx
        import torch
        from transformers import AutoTokenizer, GenerationConfig

        if not torch.cuda.is_available():
            raise RuntimeError("This experiment requires the approved CUDA GPU.")
        if version("vllm") != ENGINE_VERSION:
            raise RuntimeError("The image does not contain the pinned vLLM version.")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        generation = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        eos_ids = generation.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        if not eos_ids:
            raise ValueError("The pinned checkpoint must define EOS tokens.")
        command = [
            "vllm", "serve", MODEL_ID,
            "--revision", REVISION, "--tokenizer-revision", REVISION,
            "--served-model-name", MODEL_ID, "--host", "127.0.0.1", "--port", "8000",
            "--dtype", "float16", "--kv-cache-dtype", "float16",
            "--max-model-len", "2048", "--max-num-seqs", "2",
            "--max-num-batched-tokens", "256", "--gpu-memory-utilization", "0.5",
            "--enforce-eager", "--disable-prefix-caching", "--enable-chunked-prefill",
            "--no-async-scheduling", "--generation-config", "vllm", "--enable-request-id-headers",
        ]
        record["runtime"] = {
            "model": MODEL_ID, "revision": REVISION, "tokenizer_revision": REVISION,
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "engine": "vllm", "engine_version": ENGINE_VERSION, "image": ENGINE_IMAGE,
            "environment": environment(),
            "package_versions": {name: version(name) for name in (
                "vllm", "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "httpx",
            )},
            "gpu": torch.cuda.get_device_name(0), "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "torch_cuda_version": torch.version.cuda, "launch_command": command,
            "declared_dtype": "float16", "declared_kv_cache_dtype": "float16",
            "max_num_seqs": 2, "max_model_len": 2048, "max_num_batched_tokens": 256,
            "prefix_caching": False, "chunked_prefill": True, "async_scheduling": False,
            "eos_token_ids": eos_ids,
            "timing_boundary": "HTTP/SSE over container loopback; excludes image build, model loading and server startup.",
        }
        print("Launching:", " ".join(command), flush=True)
        process = subprocess.Popen(command, start_new_session=True)

        async def exercise():
            started = perf_counter()
            async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                while perf_counter() - started < 360:
                    if process.poll() is not None:
                        raise RuntimeError(f"vLLM exited before readiness with code {process.returncode}.")
                    try:
                        response = await client.get("http://127.0.0.1:8000/health")
                        if response.status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.5)
                else:
                    raise TimeoutError("vLLM did not become ready within 360 seconds.")
            record["server_ready_ms"] = (perf_counter() - started) * 1000
            await measure("http://127.0.0.1:8000", record, tokenizer, eos_ids)

        # Leave time for process-group cleanup before Modal's hard function timeout.
        asyncio.run(asyncio.wait_for(exercise(), timeout=480))
    except (Exception, KeyboardInterrupt) as error:
        record.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      exception_type=type(error).__name__, error_message=str(error))
        traceback.print_exc()
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    record.update(status="failed", cleanup_error="server_did_not_exit")
    return record


@app.local_entrypoint()
def main(output: str):
    from benchmarks.continuous_batching import new_result

    if Path(output).exists():
        raise FileExistsError("Choose a fresh output path; previous evidence is never overwritten.")
    record = new_result()
    record["remote_results_received"] = False
    try:
        record.update(run_benchmark.remote())
        record["remote_results_received"] = True
    except KeyboardInterrupt:
        record.update(status="interrupted", exception_type="KeyboardInterrupt")
    except Exception as error:
        record.update(status="failed", exception_type=type(error).__name__, error_message=str(error))
    write_json(output, record)
    print(json.dumps(record, indent=2))
    print(f"Saved {output}")
    if record["status"] != "completed" or not all(record["checks"].values()):
        raise SystemExit(130 if record["status"] == "interrupted" else 1)
