"""Approval-gated, offline diagnostics and vLLM checks on exactly one Modal T4.

Run with ``modal run modal_t4_lab.py --mode diagnostic`` only after operator approval.
"""

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

IMAGE = "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
ENGINE_VERSION = "0.33.0"
MODELS = {
    "small": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "large": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"),
}
app = modal.App("inference-one-t4-lab")
image = (
    modal.Image.from_registry(IMAGE)
    .entrypoint([])
    .env({
        "HF_HOME": "/opt/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1",
    })
)


def _diagnostics():
    import torch
    import vllm
    from importlib.metadata import version

    result = {
        "gpu_listing": subprocess.run(["nvidia-smi", "-L"], check=True, text=True,
                                      capture_output=True).stdout.strip(),
        "vllm_version": version("vllm"),
        "vllm_import": vllm.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "devices": [],
        "model_cache": [_cache(*spec) for spec in MODELS.values()],
    }
    if result["vllm_version"] != ENGINE_VERSION:
        raise RuntimeError(f"Expected vLLM {ENGINE_VERSION}, got {result['vllm_version']}.")
    if not result["cuda_available"]:
        raise RuntimeError("CUDA is unavailable.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}.")
    device = torch.cuda.get_device_properties(0)
    result["devices"].append({"name": device.name, "capability": list(device.major_minor) if hasattr(device, "major_minor") else [device.major, device.minor],
                              "memory_bytes": device.total_memory})
    if "T4" not in device.name:
        raise RuntimeError(f"Expected NVIDIA T4, got {device.name}.")
    return result


def _request(url, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.read().decode()


def _stop_process_group(process):
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return
    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
    raise RuntimeError(f"Process group {process.pid} remained after SIGTERM and SIGKILL.")


def _serve(mode):
    diag = _diagnostics()
    chosen = ["small", "large"] if mode == "shared" else [mode.removeprefix("single-")]
    missing = [row["model"] for row in diag["model_cache"] if row["model"] in
               [MODELS[name][0] for name in chosen] and not row["available"]]
    if missing:
        raise RuntimeError("Required offline model snapshots missing; obtain approval before any download: " + ", ".join(missing))
    processes = []
    ports = {"small": 8101, "large": 8102}
    deadline = time.monotonic() + 900
    report = {"diagnostics": diag, "mode": mode, "workers": []}
    try:
        for name in chosen:
            model, revision = MODELS[name]
            command = [
                "vllm", "serve", model, "--revision", revision,
                "--tokenizer-revision", revision, "--served-model-name", model,
                "--dtype", "float16", "--kv-cache-dtype", "float16",
                "--max-model-len", "2048", "--max-num-seqs", "2",
                "--max-num-batched-tokens", "256", "--gpu-memory-utilization", "0.5",
                "--enforce-eager", "--disable-prefix-caching", "--enable-chunked-prefill",
                "--no-async-scheduling", "--generation-config", "vllm",
                "--enable-request-id-headers", "--host", "127.0.0.1",
                "--port", str(ports[name]),
            ]
            if name == "large":
                command += ["--reasoning-parser", "qwen3",
                            "--default-chat-template-kwargs", '{"enable_thinking": false}']
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"
            process = subprocess.Popen(command, env=env, start_new_session=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            processes.append((name, model, revision, ports[name], process))
        worker_started = {row[0]: time.monotonic() for row in processes}
        pending = {row[0] for row in processes}
        while pending and time.monotonic() < deadline:
            for name, model, revision, port, process in processes:
                if name not in pending:
                    continue
                if time.monotonic() - worker_started[name] > 360:
                    raise TimeoutError(f"Worker {name} exceeded the existing 360-second startup limit.")
                if process.poll() is not None:
                    raise RuntimeError(f"Worker {name} exited during startup ({process.returncode}).")
                try:
                    status, body = _request(f"http://127.0.0.1:{port}/v1/models")
                    if status == 200:
                        response = json.loads(body)
                        if model not in [item.get("id") for item in response.get("data", [])]:
                            raise RuntimeError(f"Unexpected served model identity for {name}.")
                        pending.remove(name)
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
        if pending:
            raise TimeoutError("Workers did not become ready within the 900-second hard deadline.")
        for name, model, revision, port, process in processes:
            base = f"http://127.0.0.1:{port}/v1/chat/completions"
            common = {"model": model, "messages": [{"role": "user", "content": "Reply with the number 4."}],
                      "max_tokens": 8, "temperature": 0}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The 900-second lab deadline expired.")
            non_streaming_status, body = _request(base, common, timeout=min(180, remaining))
            if non_streaming_status != 200 or not json.loads(body).get("choices"):
                raise RuntimeError(f"Non-streaming inference failed for {name}.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The 900-second lab deadline expired.")
            streaming_status, streamed = _request(base, {**common, "stream": True}, timeout=min(180, remaining))
            if streaming_status != 200 or "data:" not in streamed:
                raise RuntimeError(f"Streaming inference failed for {name}.")
            if process.poll() is not None:
                raise RuntimeError(f"Worker {name} exited after its request checks ({process.returncode}).")
            report["workers"].append({"id": name, "model": model, "revision": revision,
                                      "pid": process.pid, "process_running": True,
                                      "endpoint": f"127.0.0.1:{port}", "ready": True,
                                      "non_streaming_status": non_streaming_status,
                                      "streaming_status": streaming_status})
        report["memory_after_requests"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader"],
            check=True, text=True, capture_output=True).stdout.strip()
        return report
    finally:
        cleanup_errors = []
        for _, _, _, _, process in processes:
            try:
                _stop_process_group(process)
            except Exception as error:
                cleanup_errors.append(f"{process.pid}: {type(error).__name__}")
        if cleanup_errors:
            raise RuntimeError("Worker process-group cleanup failed: " + ", ".join(cleanup_errors))

@app.function(image=image, gpu="T4", cpu=4, memory=8192, max_containers=1,
              min_containers=0, scaledown_window=2, timeout=958, retries=0)
def lab(mode: str):
    if mode == "diagnostic":
        return _diagnostics()
    if mode not in ("single-small", "single-large", "shared"):
        raise ValueError("mode must be diagnostic, single-small, single-large, or shared")
    return _serve(mode)


@app.local_entrypoint()
def main(mode: str = "diagnostic"):
    print(json.dumps(lab.remote(mode), indent=2))
