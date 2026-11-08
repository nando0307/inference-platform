"""Bounded real-loopback GPU routing evidence; --help needs no Modal SDK or GPU."""

import argparse
import asyncio
import json
import socket
import subprocess
from datetime import datetime, timezone
from time import perf_counter

from benchmarks.common import environment, write_json

LIMITS = {
    "smoke": {"generations": 6, "seconds": 480, "function_seconds": 600, "loads": 1},
    "lifecycle": {"generations": 32, "seconds": 780, "function_seconds": 900, "loads": 3},
}
LONG_ANSWER = "Count upward from 1 to 200, writing every integer separated by a comma. Do not summarize."


def new_result(mode):
    if mode not in LIMITS:
        raise ValueError("Choose smoke or lifecycle explicitly.")
    return {
        "experiment": "two-model-routing", "mode": mode, "status": "not_started",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "limits": {**LIMITS[mode], "max_output_tokens": 128, "max_context_tokens": 2048},
        "runtime": {}, "requests": [], "observations": [], "memory": [],
        "manager_events": [], "engine_events": [], "checks": {}, "quality": [],
        "scope": {
            "transport": "Actual HTTP/SSE over container loopback, not an ASGI test transport.",
            "timing": "Single observations only; request timing includes queueing and model switches. No medians or speedup claims.",
            "memory": "nvidia-smi whole-device used/total MiB, including all processes and CUDA overhead; not model allocation or peak memory.",
            "overlap": "Admission overlap is not proof of simultaneous GPU kernels or decoding. No GPU overlap claim is made.",
            "quality": "Frozen precision.QUALITY_CASES and score_quality; scores never gate infrastructure success.",
            "failures": "Controlled crash/load failure injections excluded to preserve the load budget; delegated to deterministic local router tests, not exercised by this GPU job.",
        },
    }


def _fail(record, error):
    record["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, asyncio.CancelledError)) else "failed"
    message = f"{type(error).__name__}: {error}"
    record["error"] = f"{record['error']}; {message}" if record.get("error") else message


def _require(record, name, condition):
    record["checks"][name] = bool(condition)
    if not condition:
        raise RuntimeError(f"Required routing check failed: {name}")


def _memory(record, label):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    devices = []
    for line in result.stdout.splitlines():
        index, uuid, used, total = (part.strip() for part in line.split(","))
        devices.append({"index": int(index), "uuid": uuid, "used_mib": int(used), "total_mib": int(total)})
    if not devices:
        raise RuntimeError("No device-level memory observation returned by nvidia-smi.")
    record["memory"].append({"label": label, "timestamp": perf_counter(), "devices": devices})


class _IdentityStream:
    """Validate wire identity before handing the same events to the strict SSE client."""

    def __init__(self, response, model, request_id):
        self.response, self.model, self.request_id = response, model, request_id

    async def aiter_lines(self):
        from benchmarks.client import StreamError, _events

        async for event in _events(self.response):
            if event != "[DONE]":
                chunk = json.loads(event)
                if not isinstance(chunk, dict) or chunk.get("model") != self.model or chunk.get("id") != f"chatcmpl-{self.request_id}":
                    raise StreamError("response_identity_mismatch")
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("reasoning") or delta.get("reasoning_content"):
                        raise StreamError("unexpected_thinking_output")
            yield f"data: {event}"
            yield ""


async def _request(client, url, record, case, *, stream=True):
    from benchmarks.client import _consume_stream, _terminal_usage, _token_ids

    if len(record["requests"]) >= record["limits"]["generations"]:
        raise RuntimeError("The hard completion-attempt budget is exhausted.")
    row = {
        "case": case["name"], "model": case["model"], "max_tokens": case["max_tokens"],
        "stream": stream, "status": "running", "request_id": None, "http_status": None,
        "started_at": perf_counter(), "first_text_ms": None, "e2e_ms": None,
        "text": "", "events": [], "prompt_token_ids": None,
        "expected_prompt": {key: case[key] for key in (
            "prompt_tokens", "prompt_token_ids", "chat_template_sha256", "formatted_prompt_sha256", "eos_token_ids",
        )},
    }
    record["requests"].append(row)
    payload = {key: case[key] for key in ("model", "messages", "max_tokens")}
    payload.update(temperature=0, top_p=1, stream=stream)
    try:
        async with client.stream("POST", url + "/v1/chat/completions", json=payload) as response:
            row.update(http_status=response.status_code, request_id=response.headers.get("x-request-id"))
            if response.status_code != 200 or not row["request_id"]:
                raise RuntimeError(f"Completion returned HTTP {response.status_code} or no request ID.")
            if stream:
                if response.headers.get("content-type", "").split(";", 1)[0] != "text/event-stream":
                    raise ValueError("Expected text/event-stream.")
                await _consume_stream(
                    _IdentityStream(response, case["model"], row["request_id"]), row, case, row["started_at"],
                )
                row["output_token_ids"] = [token for event in row["events"] for token in (event["token_ids"] or [])]
            else:
                await response.aread()
                body = response.json()
                if body.get("model") != case["model"] or body.get("id") != f"chatcmpl-{row['request_id']}":
                    raise ValueError("Nonstream response identity mismatch.")
                choice, = body["choices"]
                message = choice["message"]
                if choice.get("index") != 0 or message.get("role") != "assistant" or not isinstance(message.get("content"), str):
                    raise ValueError("Invalid nonstream assistant message.")
                if message.get("reasoning") or message.get("reasoning_content"):
                    raise ValueError("Unexpected thinking output.")
                reason = choice["finish_reason"]
                if reason not in ("stop", "length"):
                    raise ValueError("Invalid nonstream finish reason.")
                row.update(_terminal_usage(body.get("usage"), case, reason))
                row.update(status="success", text=message["content"], finish_reason=reason,
                           prompt_token_ids=_token_ids(body.get("prompt_token_ids")),
                           output_token_ids=_token_ids(choice.get("token_ids")))
            row["checks"] = {
                "wire_identity_matches": True,
                "prompt_ids_match_own_tokenizer": row["prompt_token_ids"] == case["prompt_token_ids"],
                "nonthinking": "<think>" not in row["text"] and "</think>" not in row["text"],
                "reasoning_fields_empty": True,
                "output_within_cap": 0 < row["completion_tokens"] <= case["max_tokens"],
            }
            if not all(row["checks"].values()):
                raise ValueError(f"Request checks failed: {case['name']}")
    except asyncio.CancelledError:
        row["status"] = "canceled"
        raise
    except Exception as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if row["e2e_ms"] is None:
            row["e2e_ms"] = (perf_counter() - row["started_at"]) * 1000
    return row


async def _until(predicate, *, timeout=20):
    deadline = perf_counter() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if perf_counter() >= deadline:
            raise TimeoutError("Required routing state was not observed before its bounded deadline.")
        await asyncio.sleep(0.002)


def _events(manager, name, *, model=None, request_id=None):
    return [event for event in manager.events if event["event"] == name
            and (model is None or event.get("model") == model)
            and (request_id is None or event.get("request_id") == request_id)]


def _observe(record, manager, label):
    snapshot = manager.snapshot()
    record["observations"].append({"label": label, "timestamp": perf_counter(), **snapshot})
    return snapshot


async def _smoke(client, url, record, engine, manager):
    from src.vllm_engine import HIGH_MODEL

    model = HIGH_MODEL
    rows = []
    for name, prompt, cap, stream in (
        ("nonstream_eos", "Reply with exactly one word: Hello", 64, False),
        ("stream_eos", "What is 2 + 2? Reply with only the integer.", 128, True),
        ("stream_length", LONG_ANSWER, 1, True),
    ):
        messages = [{"role": "user", "content": prompt}]
        case = {"name": name, "model": model, "messages": messages, "max_tokens": cap,
                **await engine.prepare(model, messages, cap)}
        rows.append(await _request(client, url, record, case, stream=stream))
    _require(record, f"{model}:natural_eos", any(
        row["finish_reason"] == "stop" and row["completion_tokens"] < row["max_tokens"] for row in rows[:2]
    ))
    _require(record, f"{model}:length_control", rows[2]["finish_reason"] == "length" and rows[2]["completion_tokens"] == 1)
    ready = await client.get(url + "/ready")
    snapshot = _observe(record, manager, f"smoke_ready:{model}")
    _require(record, f"{model}:ready", ready.status_code == 200 and snapshot["ready"] and snapshot["model"] == model)
    record["runtime"].setdefault("loaded_models", []).append(dict(engine.metadata))
    _memory(record, f"resident:{model}")


async def _lifecycle(client, url, record, engine, manager):
    from benchmarks.precision import QUALITY_CASES, score_quality
    from src.vllm_engine import LOW_MODEL, HIGH_MODEL

    async def case(model, name, prompt=LONG_ANSWER, cap=128):
        messages = [{"role": "user", "content": prompt}]
        return {"name": name, "model": model, "messages": messages, "max_tokens": cap,
                **await engine.prepare(model, messages, cap)}

    prepared = {}
    for model in (LOW_MODEL, HIGH_MODEL):
        prepared[model] = [await case(model, f"quality:{item['name']}", item["prompt"], 64) for item in QUALITY_CASES]
    warm = await case(LOW_MODEL, "warm_low", "Reply with exactly one word: Hello", 64)
    active_case = await case(LOW_MODEL, "low_active")
    high_case = await case(HIGH_MODEL, "high_queued")
    cancel_case = await case(LOW_MODEL, "queued_cancel")
    high_second = await case(HIGH_MODEL, "simultaneous_high",
                             "Count downward from 300 to 100, writing every integer separated by a comma. Do not summarize.")
    returning = await case(LOW_MODEL, "return_low", "Reply with exactly one word: Hello", 64)

    async def quality(model):
        for frozen, prompt in zip(QUALITY_CASES, prepared[model]):
            row = await _request(client, url, record, prompt)
            record["quality"].append({"model": model, "case": frozen["name"], "category": frozen["category"],
                                      "request_id": row["request_id"],
                                      "passed": score_quality(frozen, row["text"], row["finish_reason"])})

    await _request(client, url, record, warm)
    await _request(client, url, record, warm)
    _require(record, "warm_low_single_load", len(_events(manager, "load_started", model=LOW_MODEL)) == 1)
    low_snapshot = _observe(record, manager, "warm_low_ready")
    _require(record, "warm_low_ready", (await client.get(url + "/ready")).status_code == 200
             and low_snapshot["ready"] and low_snapshot["model"] == LOW_MODEL)
    record["runtime"].setdefault("loaded_models", []).append(dict(engine.metadata))
    _memory(record, "warm_low")
    await quality(LOW_MODEL)
    await _until(lambda: manager.snapshot()["active"] == 0 and manager.snapshot()["queued"] == 0)

    tasks = []
    try:
        low = asyncio.create_task(_request(client, url, record, active_case))
        tasks.append(low)
        await _until(lambda: manager.snapshot()["active"] == 1 and manager.snapshot()["model"] == LOW_MODEL)
        high = asyncio.create_task(_request(client, url, record, high_case))
        tasks.append(high)
        await _until(lambda: manager.snapshot()["active"] == 1 and manager.snapshot()["queued"] == 1
                     and manager.snapshot()["model"] == LOW_MODEL)
        snapshot = _observe(record, manager, "low_active_high_queued")
        _require(record, "low_active_high_queued", snapshot["active"] == 1 and snapshot["queued"] == 1
                 and snapshot["model"] == LOW_MODEL)
        queued_before = {event["request_id"] for event in _events(manager, "queued")}
        canceled = asyncio.create_task(_request(client, url, record, cancel_case))
        tasks.append(canceled)
        queued = await _until(lambda: next((event for event in _events(manager, "queued", model=LOW_MODEL)
                                           if event["request_id"] not in queued_before), None))
        cancel_id = queued["request_id"]
        snapshot = _observe(record, manager, "later_low_behind_high")
        _require(record, "cross_model_head_not_overtaken", snapshot["active"] == 1 and snapshot["queued"] == 2
                 and snapshot["model"] == LOW_MODEL and not _events(manager, "admitted", request_id=cancel_id))
        canceled.cancel()
        await asyncio.gather(canceled, return_exceptions=True)
        await _until(lambda: _events(manager, "canceled", request_id=cancel_id))
        _require(record, "queued_cancel_not_admitted", not _events(manager, "admitted", request_id=cancel_id))
        cancel_row = next(row for row in record["requests"] if row["case"] == "queued_cancel")
        cancel_row.update(request_id=cancel_id, cancellation_evidence="queued event ID; canceled before HTTP response headers")
        second = asyncio.create_task(_request(client, url, record, high_second))
        tasks.append(second)
        await asyncio.gather(low, high, second)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    low_row = next(row for row in record["requests"] if row["case"] == "low_active")
    high_rows = [row for row in record["requests"] if row["case"] in ("high_queued", "simultaneous_high")]
    released, = _events(manager, "released", request_id=low_row["request_id"])
    load, = _events(manager, "load_started", model=HIGH_MODEL)
    _require(record, "active_drained_before_high_load", released["timestamp"] <= load["timestamp"] and load["active"] == 0)
    _require(record, "simultaneous_high_single_load", len(_events(manager, "load_started", model=HIGH_MODEL)) == 1
             and all(_events(manager, "admitted", request_id=row["request_id"]) for row in high_rows))
    _require(record, "two_high_leases_admitted_together", any(
        event["active"] == 2 for row in high_rows
        for event in _events(manager, "admitted", request_id=row["request_id"])
    ))
    snapshot = _observe(record, manager, "high_ready")
    _require(record, "high_ready", (await client.get(url + "/ready")).status_code == 200
             and snapshot["ready"] and snapshot["model"] == HIGH_MODEL)
    record["runtime"]["loaded_models"].append(dict(engine.metadata))
    _memory(record, "high_resident")
    await quality(HIGH_MODEL)
    await _request(client, url, record, returning)
    await _request(client, url, record, returning)
    snapshot = _observe(record, manager, "returned_low_ready")
    _require(record, "returned_low", snapshot["model"] == LOW_MODEL and snapshot["ready"]
             and len(_events(manager, "load_started", model=LOW_MODEL)) == 2)
    record["runtime"]["loaded_models"].append(dict(engine.metadata))
    _memory(record, "returned_low")
    before = len(engine.events)
    rejected = await client.post(url + "/v1/chat/completions", json={
        "model": "unapproved/model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 1,
    })
    record["unknown_model"] = {"http_status": rejected.status_code, "body": rejected.json()}
    _require(record, "unknown_model_rejected_without_load", rejected.status_code in (400, 422) and len(engine.events) == before)
    _require(record, "all_frozen_quality_cases_recorded", len(record["quality"]) == 18 and len(QUALITY_CASES) == 9)


async def run_experiment(record, engine=None):
    """Run the real application; preserve partial rows and always stop its process group."""
    import httpx
    import uvicorn
    from src.model_router import create_app
    from src.vllm_engine import HIGH_MODEL, LOW_MODEL, MODEL_SPECS, VLLMEngine

    mode = record["mode"]
    engine = engine if engine is not None else VLLMEngine()
    app = create_app(engine)
    manager = None
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", timeout_graceful_shutdown=20))
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server_task = None
    record["status"] = "running"
    record["runtime"].update(environment=environment(), models=MODEL_SPECS, transport_url=url)

    async def exercise():
        nonlocal server_task, manager
        _memory(record, "before_model_load")
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        await _until(lambda: server.started or server_task.done())
        if server_task.done():
            await server_task
            raise RuntimeError("Loopback router exited before startup.")
        manager = app.state.manager
        async with httpx.AsyncClient(timeout=420, trust_env=False,
                                     limits=httpx.Limits(max_connections=6, max_keepalive_connections=6)) as client:
            health, ready = await client.get(url + "/health"), await client.get(url + "/ready")
            snapshot = _observe(record, manager, "initial_unloaded")
            _require(record, "initial_health_and_readiness", health.status_code == 200 and ready.status_code == 503
                     and not snapshot["ready"] and snapshot["model"] is None)
            await (_smoke if mode == "smoke" else _lifecycle)(client, url, record, engine, manager)

    try:
        await asyncio.wait_for(exercise(), timeout=LIMITS[mode]["seconds"])
        record["checks"]["internal_deadline_met"] = True
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as error:
        _fail(record, error)
    finally:
        if server_task is not None:
            server.should_exit = True
            try:
                await asyncio.wait_for(asyncio.shield(server_task), timeout=35)
            except (Exception, asyncio.CancelledError) as error:
                _fail(record, error)
                server_task.cancel()
                finished, _ = await asyncio.wait({server_task}, timeout=5)
                if not finished:
                    _fail(record, TimeoutError("Loopback server did not stop within its cancellation deadline."))
                elif not server_task.cancelled():
                    server_task.exception()
        listener.close()
        try:
            await engine.stop()
        except (Exception, asyncio.CancelledError) as error:
            _fail(record, error)
        record["manager_events"] = list(manager.events) if manager is not None else []
        record["engine_events"] = list(engine.events)
        record["runtime"]["last_engine_metadata"] = dict(engine.metadata)
        final = _observe(record, manager, "after_shutdown") if manager is not None else {}
        record["checks"]["shutdown_clean"] = (bool(final) and not engine.is_alive() and engine.model is None
                                               and final["active"] == 0 and final["queued"] == 0 and not final["accepting"])
        try:
            _memory(record, "after_process_group_cleanup")
        except Exception as error:
            _fail(record, error)
        rows = record["requests"]
        record["checks"]["request_budget_respected"] = len(rows) <= LIMITS[mode]["generations"] and all(row["max_tokens"] <= 128 for row in rows)
        record["checks"]["requests_valid"] = bool(rows) and all(
            (row["status"] == "success" and all(row.get("checks", {}).values()))
            or (mode == "lifecycle" and row["case"] == "queued_cancel" and row["status"] == "canceled"
                and record["checks"].get("queued_cancel_not_admitted") is True) for row in rows
        )
        ids = [row["request_id"] for row in rows if row["request_id"]]
        record["checks"]["distinct_request_identities"] = len(ids) == len(rows) and len(set(ids)) == len(ids)
        loads = [event for event in engine.events if event["event"] == "load_started"]
        unloads = [event for event in engine.events if event["event"] == "unload_finished"]
        expected_models = [LOW_MODEL, HIGH_MODEL, LOW_MODEL] if mode == "lifecycle" else [HIGH_MODEL]
        record["checks"]["bounded_model_loads"] = [event["model"] for event in loads] == expected_models
        record["checks"]["single_resident_transition_order"] = (
            len(unloads) == len(loads) and all(
                unloaded["model"] == loaded["model"] and unloaded["at"] >= loaded["at"]
                for loaded, unloaded in zip(loads, unloads)
            ) and all(unloads[index - 1]["at"] <= loaded["at"] for index, loaded in enumerate(loads) if index)
        )
        record["checks"]["all_process_groups_stopped"] = len(unloads) == len(loads) and all(
            event.get("stopped") is True and event.get("forced") is False and event.get("error") is None for event in unloads
        )
        manager_events = record["manager_events"]
        record["checks"]["bounded_admission"] = bool(manager_events) and all(
            0 <= event["active"] <= 2 and 0 <= event["queued"] <= 2 for event in manager_events
        )
        record["checks"]["load_transitions_drained"] = all(
            event["active"] == 0 for event in manager_events if event["event"] == "load_started"
        )
        record["checks"]["all_passed"] = all(record["checks"].values())
        if record["status"] == "running":
            record["status"] = "completed" if record["checks"]["all_passed"] else "failed"
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=tuple(LIMITS), help="Explicit phase; lifecycle requires separately approved successful smoke.")
    parser.add_argument("--output", required=True, help="Fresh JSON artifact path; existing evidence is never overwritten.")
    args = parser.parse_args(argv)
    from pathlib import Path

    if Path(args.output).exists():
        raise FileExistsError("Choose a fresh output path.")
    record = new_result(args.mode)
    try:
        asyncio.run(run_experiment(record))
    except (Exception, KeyboardInterrupt) as error:
        _fail(record, error)
    finally:
        write_json(args.output, record)
    print(json.dumps({"status": record["status"], "checks": record["checks"], "output": args.output}, indent=2))
    return 0 if record["status"] == "completed" else 130 if record["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
