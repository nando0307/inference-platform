"""Matched HTTP observations for the unchanged Transformers API and vLLM."""

import argparse
import asyncio
import hashlib
import json
import math
import re
import signal
import socket
from contextlib import asynccontextmanager, contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from benchmarks.common import environment
from benchmarks.continuous_batching import _prompt, _scenario
from benchmarks.workloads import MODEL_ID, SHORT_PROMPT


def new_result():
    return {
        "schema_version": 1,
        "kind": "engine_comparison",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_started",
        "runtime": {},
        "backends": [],
        "checks": {},
        "error": None,
    }


def new_backend_result(backend):
    if backend not in ("transformers", "vllm"):
        raise ValueError("Backend must be transformers or vllm.")
    return {
        "backend": backend,
        "status": "not_started",
        "runtime": {},
        "workload": None,
        "scenarios": [],
        "checks": {},
        "error": None,
        "timing_limits": (
            "One observation per measured scenario, after a matching discarded warmup. "
            "Client wall-clock first text and terminal SSE latency include queueing, "
            "tokenization, streaming and transport. Chunk gaps are not token latency. "
            "HTTP overlap is not an engine scheduler trace. No CUDA kernel timings, "
            "percentiles, output-ID equality or quality claims."
        ),
    }


def _failed(record, error):
    record["status"] = "failed"
    record["checks"]["all_passed"] = False
    message = f"{type(error).__name__}: {error}"
    if not record["error"]:
        record["error"] = message
    elif not record["error"].endswith(message):
        record["error"] += "; " + message


async def measure_backend(url, record, tokenizer, eos_token_ids, *, fixed_scope=None):
    """Run fourteen requests using the existing strict SSE parser and scenario runner."""
    import httpx

    record["status"] = "running"
    try:
        if not eos_token_ids or any(type(token) is not int or token < 0 for token in eos_token_ids):
            raise ValueError("Pinned generation config must supply nonnegative integer EOS IDs.")
        if record["backend"] == "transformers" and fixed_scope is None:
            raise ValueError("Transformers fixed workloads require a scoped generation control.")
        prompt = _prompt(tokenizer, SHORT_PROMPT)
        cases = [
            {**prompt, "name": name, "max_tokens": cap}
            for name, cap in (("A", 16), ("B", 128), ("C", 32))
        ]
        natural = {
            **_prompt(tokenizer, "Reply with exactly one word: Hello"),
            "name": "natural_eos", "max_tokens": 32,
        }
        if any(case["prompt_tokens"] + case["max_tokens"] > 2048 for case in [*cases, natural]):
            raise ValueError("Comparison workload exceeds the 2048-token context budget.")
        workload = {
            "model": MODEL_ID,
            "model_revision": record["runtime"].get("revision"),
            "tokenizer_revision": record["runtime"].get("tokenizer_revision"),
            "generation": {"temperature": 0, "top_p": 1},
            "context_limit": 2048,
            "eos_token_ids": sorted(eos_token_ids),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "cases": cases,
            "natural_eos_case": natural,
        }
        record["workload"] = workload
        limits = httpx.Limits(max_connections=2, max_keepalive_connections=2)
        async with httpx.AsyncClient(timeout=120, limits=limits, trust_env=False) as client:
            for name, selected, refill, fixed in (
                ("warmup_sequential", cases, False, True),
                ("warmup_refill", cases, True, True),
                ("sequential", cases, False, True),
                ("refill", cases, True, True),
                ("warmup_natural_eos", [natural], False, False),
                ("natural_eos", [natural], False, False),
            ):
                scenario = {
                    "name": name, "discarded": name.startswith("warmup_"),
                    "admission_plan": (
                        "A and B at scenario start; C immediately after A finishes"
                        if refill else "One request at a time, in listed order"
                    ),
                    "planned_cases": [case["name"] for case in selected],
                }
                record["scenarios"].append(scenario)
                with fixed_scope() if fixed and fixed_scope is not None else nullcontext():
                    await _scenario(
                        client, url, workload, selected, scenario,
                        refill=refill, fixed=fixed, vllm_options=record["backend"] == "vllm",
                    )
        scenarios = {scenario["name"]: scenario for scenario in record["scenarios"]}
        record["diagnostics"] = {
            "refill_demonstrated": scenarios["refill"]["checks"]["refill_demonstrated"],
            "refill_required_for_acceptance": False,
            "natural_control_is_quality_test": False,
        }
        record["checks"].update(
            requests_valid=all(
                row["checks"]["valid_usage_and_terminal_stream"]
                for scenario in record["scenarios"] for row in scenario["rows"]
            ),
            fixed_completion_counts_match=all(
                row["checks"]["fixed_completion_count_matches_cap"]
                for scenario in record["scenarios"] if "natural_eos" not in scenario["name"]
                for row in scenario["rows"]
            ),
            natural_eos_demonstrated=all(
                scenarios[name]["checks"]["early_eos"]
                for name in ("warmup_natural_eos", "natural_eos")
            ),
        )
        record["checks"]["all_passed"] = all(
            value for key, value in record["checks"].items() if key != "all_passed"
        )
        if not record["checks"]["all_passed"]:
            raise RuntimeError("Required backend acceptance checks failed.")
        record["status"] = "completed"
    except BaseException as error:
        _failed(record, error)
        raise
    return record


@contextmanager
def fixed_generation(runtime):
    """Suppress EOS only on the fresh production per-call config, never model defaults."""
    original = runtime.model.generate

    def fixed(*args, **kwargs):
        config = kwargs["generation_config"]
        if config is runtime.model.generation_config:
            raise ValueError("Fixed generation requires a fresh per-call config.")
        previous = config.min_new_tokens
        config.min_new_tokens = config.max_new_tokens
        try:
            return original(*args, **kwargs)
        finally:
            config.min_new_tokens = previous

    with patch.object(runtime.model, "generate", fixed):
        yield


@contextmanager
def _observe_baseline(runtime, record):
    original = runtime.model.generate
    observations = record.setdefault("generation_observations", [])

    def observed(*args, **kwargs):
        scenario = record["scenarios"][-1]
        config = kwargs["generation_config"]
        row = {
            "scenario": scenario["name"],
            "discarded": scenario["discarded"],
            "generation_controls": {
                key: getattr(config, key) for key in (
                    "max_new_tokens", "min_new_tokens", "do_sample", "temperature", "top_p",
                    "top_k", "use_cache", "num_beams", "num_return_sequences",
                    "repetition_penalty", "eos_token_id", "pad_token_id",
                )
            },
        }
        observations.append(row)
        if scenario["discarded"]:
            workload = record["workload"]
            case = (
                workload["natural_eos_case"] if "natural_eos" in scenario["name"]
                else next(case for case in workload["cases"] if case["max_tokens"] == config.max_new_tokens)
            )
            row["case"] = case["name"]
            row["input_ids"] = kwargs["input_ids"].detach().cpu().tolist()
            row["input_ids_match_workload"] = row["input_ids"] == [case["prompt_token_ids"]]
            if not row["input_ids_match_workload"]:
                raise ValueError("Observed baseline model input IDs differ from the workload.")
        return original(*args, **kwargs)

    with patch.object(runtime.model, "generate", observed):
        yield


async def _wait_ready(url, server_task):
    import httpx

    async with httpx.AsyncClient(timeout=1, trust_env=False) as client:
        while True:
            if server_task.done():
                server_task.result()
                raise RuntimeError("Baseline server exited before readiness.")
            try:
                response = await client.get(url + "/ready")
                if response.status_code == 200 and response.json().get("ready") is True:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.05)


async def _stop_server(server, task, app):
    """Stop admission, signal producer cancellation, then drain HTTP and lifespan."""
    state = app.state
    state.shutting_down = True
    from src.api import cancel_inference

    for job in tuple(getattr(state, "pending_jobs", ())):
        cancel_inference(app, job, "shutdown")
    if getattr(state, "active_job", None) is not None:
        cancel_inference(app, state.active_job, "shutdown")
    server.should_exit = True
    try:
        await asyncio.wait_for(asyncio.shield(task), 8)
    except BaseException:
        server.force_exit = True
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=2)
        if done:
            await asyncio.gather(task, return_exceptions=True)
        raise
    if (
        getattr(state, "executor", None) is not None
        or getattr(state, "producer_futures", ())
        or getattr(state, "runtime", None) is not None
    ):
        raise RuntimeError("Baseline server did not finish inference cleanup.")


async def exercise_baseline(runtime, record, *, port=8000, timeout=120):
    """Exercise a caller-owned runtime through real loopback HTTP, including CPU smoke runs."""
    import torch
    import uvicorn
    from src import api

    if record["backend"] != "transformers":
        raise ValueError("Baseline requires a transformers backend record.")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite.")
    metadata = record["runtime"]
    metadata.update(
        model=MODEL_ID,
        device=str(runtime.model.device), dtype=str(runtime.model.dtype),
        declared_dtype="float16" if runtime.model.dtype == torch.float16 else str(runtime.model.dtype).removeprefix("torch."),
        tokenizer_revision=metadata.get("revision"),
        chat_template_sha256=hashlib.sha256(runtime.tokenizer.chat_template.encode()).hexdigest(),
        eos_token_ids=sorted(runtime.eos_token_ids), context_limit=runtime.context_limit,
        attention_implementation=getattr(runtime.model.config, "_attn_implementation", None),
        environment=environment(),
        torch_num_threads=torch.get_num_threads(),
        torch_num_interop_threads=torch.get_num_interop_threads(),
        workers=1, model_concurrency=1, queue_capacity=api.QUEUE_CAPACITY,
        input_id_evidence="Observed model.generate input_ids in discarded warmups, not HTTP fields.",
        startup_boundary="Server construction to /ready; model load recorded separately; excludes imports.",
        cuda_runtime=torch.version.cuda,
        gpu_name=(torch.cuda.get_device_name(runtime.model.device) if runtime.model.device.type == "cuda" else None),
    )
    record["status"] = "running"
    original_lifespan = api.app.router.lifespan_context

    @asynccontextmanager
    async def injected_lifespan(app):
        with patch.object(api, "load_runtime", lambda: runtime):
            async with original_lifespan(app):
                yield

    started = perf_counter()
    server_task = None
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
            listener.listen(128)
            listener.setblocking(False)
            url = f"http://127.0.0.1:{listener.getsockname()[1]}"
            metadata["url"] = url
            server = uvicorn.Server(uvicorn.Config(
                api.app, host="127.0.0.1", port=listener.getsockname()[1],
                workers=1, access_log=False, log_level="warning", lifespan="on",
                timeout_graceful_shutdown=5,
            ))
            # The disposable CLI owns signals so interruptions still persist its record.
            with (
                patch.object(api.app.router, "lifespan_context", injected_lifespan),
                patch.object(server, "capture_signals", nullcontext),
                _observe_baseline(runtime, record),
            ):
                server_task = asyncio.create_task(server.serve(sockets=[listener]))
                try:
                    async with asyncio.timeout(timeout):
                        await _wait_ready(url, server_task)
                        metadata["startup_ms"] = (perf_counter() - started) * 1000
                        await measure_backend(
                            url, record, runtime.tokenizer, runtime.eos_token_ids,
                            fixed_scope=lambda: fixed_generation(runtime),
                        )
                        warmups = [row for row in record["generation_observations"] if row["discarded"]]
                        record["checks"]["warmup_input_ids_match"] = (
                            len(warmups) == 7 and all(row["input_ids_match_workload"] for row in warmups)
                        )
                finally:
                    record["checks"]["server_stopped_cleanly"] = False
                    await _stop_server(server, server_task, api.app)
                    record["checks"]["server_stopped_cleanly"] = True
            record["checks"]["all_passed"] = all(
                value for key, value in record["checks"].items() if key != "all_passed"
            )
            if not record["checks"]["all_passed"]:
                raise RuntimeError("Required baseline acceptance checks failed.")
            record["status"] = "completed"
        except BaseException as error:
            _failed(record, error)
            raise
        finally:
            metadata["exercise_ms"] = (perf_counter() - started) * 1000
    return record


def run_baseline(revision, *, timeout, port=8000, record=None):
    """Load the pinned production runtime, then move it only here to CUDA FP16."""
    record = new_backend_result("transformers") if record is None else record
    record["runtime"].update(revision=revision, tokenizer_revision=revision, declared_dtype="float16")
    started = perf_counter()
    try:
        import torch
        from src.inference import load_runtime

        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Revision must be a full 40-character commit hash.")
        if not torch.cuda.is_available():
            raise RuntimeError("The baseline CLI requires CUDA FP16; no CPU fallback is permitted.")
        runtime = load_runtime(revision=revision)
        runtime.model.to(device="cuda", dtype=torch.float16)
        if runtime.model.device.type != "cuda" or runtime.model.dtype != torch.float16:
            raise RuntimeError("Baseline model is not actually CUDA FP16.")
        record["runtime"]["load_ms"] = (perf_counter() - started) * 1000
        remaining = timeout - (perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError("Baseline deadline expired during model loading.")
        asyncio.run(exercise_baseline(runtime, record, port=port, timeout=remaining))
    except BaseException as error:
        _failed(record, error)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", required=True)
    parser.add_argument("--output", required=True, help="Exclusive new backend JSON output.")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    record = new_backend_result("transformers")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with output.open("x", encoding="utf-8") as destination:
            # Even a hard outer-process deadline leaves an explicit incomplete record.
            destination.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
            destination.flush()
            try:
                run_baseline(args.revision, timeout=args.timeout, port=args.port, record=record)
            finally:
                destination.seek(0)
                destination.write(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                destination.truncate()
                destination.flush()
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0 if record["status"] == "completed" and record["checks"].get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
