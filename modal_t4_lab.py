"""Approval-gated cache population and vLLM checks on exactly one Modal T4.

Run with ``modal run modal_t4_lab.py --mode diagnostic`` only after operator approval.
"""

import json
import os
import signal
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal

from modal_hf_cache import MODELS as PINNED_MODELS

# Modal needs an unversioned `python`; alias vLLM's existing interpreter.
PYTHON_ALIAS_SETUP = 'RUN ln -s "$(command -v python3)" "$(dirname "$(command -v python3)")/python"'

IMAGE = "vllm/vllm-openai@sha256:91ee972ab751fb7180bc71c175865e36b091686600dca783d52b2511399810b3"
ENGINE_VERSION = "0.33.0"
MODEL_REPOS = {
    "small": "Qwen/Qwen2.5-0.5B-Instruct",
    "large": "Qwen/Qwen3-1.7B",
}
MODELS = {name: (repo, PINNED_MODELS[repo]) for name, repo in MODEL_REPOS.items()}
cache_volume = modal.Volume.from_name("inference-huggingface-cache", version=2)


app = modal.App("inference-one-t4-lab")
image = (
    modal.Image.from_registry(
        IMAGE,
        setup_dockerfile_commands=[PYTHON_ALIAS_SETUP],
    )
    .entrypoint([])
    .env({
        "HF_HOME": "/opt/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_NO_USAGE_STATS": "1",
    })
    .add_local_file(Path(__file__).parent / "modal_hf_cache.py", "/root/modal_hf_cache.py")
)


def _cache(model_path, revision):
    from modal_hf_cache import HF_HOME, MANIFEST, snapshot_path, verify_snapshot

    try:
        manifest = json.loads(MANIFEST.read_text())
        entry = manifest[model_path]
        if entry["revision"] != revision:
            raise ValueError("revision mismatch")
        verify_snapshot(snapshot_path(HF_HOME, model_path, revision), HF_HOME, entry["paths"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return {"model": model_path, "available": False}
    return {"model": model_path, "available": True}


def _populate_cache():
    from modal_hf_cache import HF_HOME, MANIFEST, MODELS as PINNED_MODELS
    from modal_hf_cache import snapshot_path, verify_manifest, verify_snapshot

    try:
        verify_manifest(HF_HOME)
        return
    except RuntimeError:
        pass

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    manifest = {}
    for repo, revision in PINNED_MODELS.items():
        expected = sorted(api.list_repo_files(repo, revision=revision))
        snapshot_download(repo, revision=revision, cache_dir=str(HF_HOME / "hub"))
        paths = verify_snapshot(snapshot_path(HF_HOME, repo, revision), HF_HOME, expected)
        manifest[repo] = {"revision": revision, "paths": paths}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    cache_volume.commit()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    verify_manifest(HF_HOME)



def _diagnostics():
    import torch
    import vllm
    from importlib.metadata import version

    result = {
        "python_executable": os.path.realpath(os.sys.executable),
        "python_version": os.sys.version,
        "path": os.environ.get("PATH", ""),
        "pip_executable": shutil.which("pip"),
        "package_paths": {"torch": torch.__file__, "vllm": vllm.__file__},
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
    exc = None
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
            try:
                non_streaming_status, body = _request(base, common, timeout=min(180, remaining))
            except (urllib.error.URLError, TimeoutError) as e:
                raise RuntimeError(f"Inference request failed for {name}") from e
            if non_streaming_status != 200 or not json.loads(body).get("choices"):
                raise RuntimeError(f"Non-streaming inference failed for {name}.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("The 900-second lab deadline expired.")
            try:
                streaming_status, streamed = _request(base, {**common, "stream": True}, timeout=min(180, remaining))
            except (urllib.error.URLError, TimeoutError) as e:
                raise RuntimeError(f"Streaming inference request failed for {name}") from e
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
    except BaseException as e:
        exc = e
    cleanup_errors = []
    for _, _, _, _, process in processes:
        try:
            _stop_process_group(process)
        except Exception as error:
            cleanup_errors.append(f"{process.pid}: {type(error).__name__}")
    if cleanup_errors:
        msg = "Worker process-group cleanup failed: " + ", ".join(cleanup_errors)
        if exc:
            raise RuntimeError(msg) from exc
        raise RuntimeError(msg)
    if exc:
        raise exc
    return report

@app.function(image=image, gpu="T4", cpu=4, memory=8192, max_containers=1,
              min_containers=0, scaledown_window=2, timeout=958, retries=0,
              volumes={"/opt/huggingface": cache_volume})
def lab(mode: str):
    if mode not in ("diagnostic", "single-small", "single-large", "shared"):
        raise ValueError("mode must be diagnostic, single-small, single-large, or shared")
    _populate_cache()
    if mode == "diagnostic":
        return _diagnostics()
    return _serve(mode)


@app.local_entrypoint()
def main(mode: str = "diagnostic"):
    print(json.dumps(lab.remote(mode), indent=2))
