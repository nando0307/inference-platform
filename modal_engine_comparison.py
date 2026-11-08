"""Compare the existing Transformers server and vLLM in one bounded T4 run."""

import json
from pathlib import Path
from time import perf_counter

import modal

from benchmarks.common import write_json
from benchmarks.workloads import MODEL_ID
from modal_continuous_batching import ENGINE_IMAGE, ENGINE_VERSION, image as engine_image
from modal_inference import REVISION
from src.vllm_engine import stop_process_group


app = modal.App("inference-qwen-engine-comparison")
image = engine_image
# The inherited image already includes common, workloads, client, continuous_batching,
# and modal_inference. Mount only the additional source files, never saved results.
for relative in (
    "src/api.py", "src/inference.py", "src/streaming.py", "src/vllm_engine.py",
    "benchmarks/engine_comparison.py", "modal_continuous_batching.py",
    "modal_engine_comparison.py",
):
    image = image.add_local_file(Path(__file__).parent / relative, f"/root/{relative}")


def _remaining(deadline):
    seconds = deadline - perf_counter()
    if seconds <= 0:
        raise TimeoutError("The shared 480-second comparison deadline expired.")
    return seconds


def _fail(record, error):
    message = f"{type(error).__name__}: {error}"
    record["error"] = f"{record['error']}; {message}" if record.get("error") else message
    record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"


def _required_checks_pass(record):
    checks = record.get("checks")
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        return False
    if record.get("backend") in ("transformers", "vllm"):
        required = (
            "requests_valid", "fixed_completion_counts_match",
            "natural_eos_demonstrated", "all_passed",
        )
        if record["backend"] == "transformers":
            required += ("warmup_input_ids_match", "server_stopped_cleanly")
        return all(checks.get(key) is True for key in required)
    return True




def _run_baseline(record, *, deadline, output):
    """Keep child evidence even on failure; return only after group cleanup."""
    import subprocess
    import sys

    from benchmarks.engine_comparison import new_backend_result

    backend = new_backend_result("transformers")
    record["backends"].append(backend)
    process = None
    failure = None
    cleanup = {"stopped": True, "forced": False, "error": None}
    command = None
    started = perf_counter()
    try:
        # Leave the child time to drain its server and write partial evidence.
        timeout = _remaining(deadline) - 15
        if timeout <= 0:
            raise TimeoutError("No time remains for baseline execution and cleanup.")
        command = [
            sys.executable, "-m", "benchmarks.engine_comparison", "--baseline",
            "--output", str(output), "--revision", REVISION,
            "--timeout", str(timeout), "--port", "8000",
        ]
        print("Launching:", " ".join(command), flush=True)
        process = subprocess.Popen(command, start_new_session=True)
        process.wait(timeout=_remaining(deadline))
    except (Exception, KeyboardInterrupt) as error:
        failure = error
    finally:
        if process is not None:
            cleanup = stop_process_group(process)
        try:
            with Path(output).open(encoding="utf-8") as source:
                child = json.load(source)
            if not isinstance(child, dict) or child.get("backend") != "transformers":
                raise ValueError("Baseline child returned an invalid backend record.")
            if not isinstance(child.get("runtime"), dict) or not isinstance(child.get("checks"), dict):
                raise ValueError("Baseline child returned invalid runtime or checks.")
            backend.update(child)
        except (OSError, ValueError, TypeError) as error:
            _fail(backend, error)
        backend["runtime"].update(image=ENGINE_IMAGE, launch_command=command)
        backend["process_elapsed_ms"] = (perf_counter() - started) * 1000
        backend["process_cleanup"] = cleanup
        backend["checks"]["process_stopped_cleanly"] = (
            cleanup["stopped"] and not cleanup["forced"] and cleanup["error"] is None
        )
        if failure is not None:
            _fail(backend, failure)
        if process is not None and process.returncode != 0:
            _fail(backend, RuntimeError(f"Baseline subprocess exited with code {process.returncode}."))
        if backend["status"] != "completed" or not _required_checks_pass(backend):
            if backend.get("error") is None:
                _fail(backend, RuntimeError("Baseline did not pass its required checks."))
            backend["checks"]["all_passed"] = False
        # Save the collected child record, including parent-observed cleanup, before vLLM.
        write_json(Path(output).with_name("baseline-collected.json"), backend)
    if isinstance(failure, KeyboardInterrupt):
        raise failure
    if not backend["checks"]["process_stopped_cleanly"]:
        raise RuntimeError("Baseline process-group cleanup failed; vLLM will not be started.")
    if backend["status"] != "completed" or not _required_checks_pass(backend):
        raise RuntimeError(f"Baseline failed; vLLM will not be started. {backend.get('error')}")
    return backend


def _comparison_checks(record):
    backends = record.get("backends", [])
    ordered = isinstance(backends, list) and len(backends) == 2 and all(
        isinstance(backend, dict) and backend.get("backend") == name
        for backend, name in zip(backends, ("transformers", "vllm"))
    )
    checks = {
        "both_backends_recorded": ordered,
        "both_backends_passed": False,
        "workloads_match": False,
        "checkpoint_and_tokenizer_match": False,
        "fp16_cuda_controls_match": False,
        "same_pinned_image": False,
        "processes_stopped_cleanly": False,
    }
    if not ordered:
        return checks
    baseline, engine = backends
    runtimes = [backend.get("runtime", {}) for backend in backends]
    checks["both_backends_passed"] = all(
        backend.get("status") == "completed" and _required_checks_pass(backend)
        for backend in backends
    )
    workload = baseline.get("workload")
    checks["workloads_match"] = (
        isinstance(workload, dict) and bool(workload) and workload == engine.get("workload")
        and workload.get("model") == MODEL_ID and workload.get("context_limit") == 2048
        and workload.get("generation") == {"temperature": 0, "top_p": 1}
        and workload.get("model_revision") == REVISION
        and workload.get("tokenizer_revision") == REVISION
    )
    checks["checkpoint_and_tokenizer_match"] = all(
        runtime.get("model") == MODEL_ID and runtime.get("revision") == REVISION
        and runtime.get("tokenizer_revision") == REVISION
        and bool(runtime.get("chat_template_sha256"))
        and bool(runtime.get("eos_token_ids"))
        for runtime in runtimes
    ) and all(
        runtimes[0].get(key) == runtimes[1].get(key) == (workload or {}).get(key)
        for key in ("chat_template_sha256", "eos_token_ids")
    )
    checks["fp16_cuda_controls_match"] = all(
        runtime.get("declared_dtype") == "float16"
        and runtime.get("device") in ("cuda", "cuda:0")
        for runtime in runtimes
    ) and str(runtimes[0].get("dtype")).removeprefix("torch.") == "float16"
    checks["same_pinned_image"] = all(runtime.get("image") == ENGINE_IMAGE for runtime in runtimes)
    checks["processes_stopped_cleanly"] = all(
        backend.get("process_cleanup", {}).get("stopped") is True
        and backend["process_cleanup"].get("forced") is False
        and backend["process_cleanup"].get("error") is None
        for backend in backends
    )
    return checks


@app.function(
    image=image, gpu="T4", cpu=4, memory=8192,
    max_containers=1, min_containers=0, scaledown_window=2, timeout=600, retries=0,
)
def run_benchmark():
    import asyncio
    import hashlib
    import subprocess
    import tempfile
    import traceback
    from importlib.metadata import version

    from benchmarks.common import environment
    from benchmarks.engine_comparison import measure_backend, new_backend_result, new_result

    record = new_result()
    record["status"] = "running"
    started = perf_counter()
    deadline = started + 480
    process = None
    engine = None
    try:
        import httpx
        import torch
        from transformers import AutoTokenizer, GenerationConfig

        record["runtime"] = {
            "image": ENGINE_IMAGE, "engine_version": ENGINE_VERSION,
            "model": MODEL_ID, "revision": REVISION, "tokenizer_revision": REVISION,
            "environment": environment(),
            "package_versions": {name: version(name) for name in (
                "vllm", "torch", "transformers", "tokenizers", "huggingface-hub",
                "safetensors", "httpx", "fastapi", "uvicorn", "pydantic",
            )},
            "resource_limits": {
                "gpu": "T4", "cpu": 4, "memory_mib": 8192, "max_containers": 1,
                "min_containers": 0, "scaledown_window_seconds": 2,
                "function_timeout_seconds": 600, "retries": 0,
            },
            "overall_timeout_seconds": 480, "vllm_readiness_timeout_seconds": 360,
            "backend_order": ["transformers", "vllm"],
            "timing_boundary": (
                "HTTP/SSE over container loopback; excludes image build, model loading, "
                "server startup, warmup, and process cleanup. Baseline serialization is expected."
            ),
        }
        if not torch.cuda.is_available():
            raise RuntimeError("This comparison requires the approved CUDA GPU.")
        if version("vllm") != ENGINE_VERSION:
            raise RuntimeError("The image does not contain the pinned vLLM version.")
        record["runtime"].update(
            gpu=torch.cuda.get_device_name(0),
            cuda_capability=list(torch.cuda.get_device_capability(0)),
            torch_cuda_version=torch.version.cuda,
        )
        with tempfile.TemporaryDirectory(prefix="engine-comparison-") as directory:
            _run_baseline(record, deadline=deadline, output=Path(directory) / "baseline.json")
            _remaining(deadline)
            engine = new_backend_result("vllm")
            record["backends"].append(engine)
            tokenizer_started = perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
            generation = GenerationConfig.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
            eos_ids = generation.eos_token_id
            eos_ids = sorted([eos_ids] if isinstance(eos_ids, int) else list(eos_ids or []))
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
            engine["runtime"] = {
                **record["runtime"], "engine": "vllm", "device": "cuda",
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "launch_command": command, "declared_dtype": "float16",
                "declared_kv_cache_dtype": "float16", "max_num_seqs": 2,
                "max_model_len": 2048, "max_num_batched_tokens": 256,
                "gpu_memory_utilization": 0.5, "enforce_eager": True,
                "prefix_caching": False, "chunked_prefill": True, "async_scheduling": False,
                "generation_config": "vllm", "request_id_headers": True,
                "eos_token_ids": sorted(eos_ids),
                "tokenizer_load_ms": (perf_counter() - tokenizer_started) * 1000,
                "startup_ms": None,
            }
            _remaining(deadline)
            print("Launching:", " ".join(command), flush=True)
            server_started = perf_counter()
            process = subprocess.Popen(command, start_new_session=True)

            async def exercise():
                readiness_deadline = min(deadline, server_started + 360)
                async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError(f"vLLM exited before readiness with code {process.returncode}.")
                        remaining = readiness_deadline - perf_counter()
                        if remaining <= 0:
                            raise TimeoutError("vLLM readiness exceeded its bounded deadline.")
                        try:
                            response = await client.get(
                                "http://127.0.0.1:8000/health", timeout=min(2, remaining),
                            )
                            if response.status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        await asyncio.sleep(min(0.5, max(0, readiness_deadline - perf_counter())))
                engine["runtime"]["startup_ms"] = (perf_counter() - server_started) * 1000
                await measure_backend("http://127.0.0.1:8000", engine, tokenizer, eos_ids)

            remaining = _remaining(deadline)
            asyncio.run(asyncio.wait_for(exercise(), timeout=remaining))
    except (Exception, KeyboardInterrupt) as error:
        if engine is not None:
            _fail(engine, error)
        _fail(record, error)
        traceback.print_exc()
    finally:
        if process is not None:
            cleanup = stop_process_group(process)
            engine["process_cleanup"] = cleanup
            engine["checks"]["process_stopped_cleanly"] = (
                cleanup["stopped"] and not cleanup["forced"] and cleanup["error"] is None
            )
            if not engine["checks"]["process_stopped_cleanly"]:
                _fail(engine, RuntimeError(f"vLLM process-group cleanup failed: {cleanup}"))
                engine["checks"]["all_passed"] = False
        record["runtime"]["elapsed_ms"] = (perf_counter() - started) * 1000
        record["checks"] = _comparison_checks(record)
        record["checks"]["shared_deadline_met"] = perf_counter() <= deadline
        if record["status"] == "running":
            if _required_checks_pass(record):
                record["status"] = "completed"
            else:
                _fail(record, RuntimeError("Comparison required checks failed; inspect backend evidence."))
    return record


@app.local_entrypoint()
def main(output: str):
    from benchmarks.engine_comparison import new_result

    if Path(output).exists() or Path(output).is_symlink():
        raise FileExistsError("Choose a fresh output path; previous evidence is never overwritten.")
    record = new_result()
    record["remote_results_received"] = False
    try:
        record.update(run_benchmark.remote())
        record["remote_results_received"] = True
        if record.get("status") == "completed" and (
            not _required_checks_pass(record) or not all(_comparison_checks(record).values())
        ):
            _fail(record, RuntimeError("Remote comparison has empty or failed required checks."))
    except (Exception, KeyboardInterrupt) as error:
        _fail(record, error)
    write_json(output, record)
    print(json.dumps(record, indent=2))
    print(f"Saved {output}")
    if record["status"] != "completed" or not _required_checks_pass(record):
        raise SystemExit(130 if record["status"] == "interrupted" else 1)
