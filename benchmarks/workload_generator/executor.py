"""Open-loop asynchronous replay against the public inference API."""

import asyncio
import json
from pathlib import Path
from time import perf_counter

import httpx


from benchmarks.client import _events as _sse_events
async def _request(client, endpoint, record, origin, *, cancel_after_s=None):
    started = perf_counter()
    result = {"result_schema_version": 1, "request_id": record["request_id"], "trace_id": record.get("trace_id"), "scenario_id": record["scenario_id"], "phase_id": record["phase_id"], "model": record["model"], "scheduled_offset_s": record["scheduled_offset_s"], "actual_submit_offset_s": started - origin, "scheduling_delay_ms": (started - origin - record["scheduled_offset_s"]) * 1000, "status": "error", "http_status": None, "error_code": None, "first_event_ms": None, "first_text_ms": None, "completion_ms": None, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    payload = {"model": record["model"], "messages": record["messages"], "max_tokens": record["max_tokens"], "temperature": record["temperature"], "top_p": record["top_p"], "stream": record["stream"]}
    try:
        async with asyncio.timeout(record["request_timeout_s"]):
            async with client.stream("POST", endpoint.rstrip("/") + "/v1/chat/completions", json=payload) as response:
                result["http_status"] = response.status_code
                result["request_id"] = response.headers.get("x-request-id", result["request_id"])
                if response.status_code != 200:
                    await response.aread()
                    try:
                        body = response.json()
                        result["error_code"] = body.get("error", {}).get("code", "http_error")
                    except ValueError:
                        result["error_code"] = "http_error"
                    result["status"] = "rejected" if response.status_code == 429 else "error"
                    return result
                if not record["stream"]:
                    body = await response.aread()
                    payload = json.loads(body)
                    result.update({"status": "success", "completion_ms": (perf_counter() - started) * 1000, "prompt_tokens": payload.get("usage", {}).get("prompt_tokens"), "completion_tokens": payload.get("usage", {}).get("completion_tokens"), "total_tokens": payload.get("usage", {}).get("total_tokens")})
                    return result
                terminal = False
                async for event in _sse_events(response):
                    now_ms = (perf_counter() - started) * 1000
                    if terminal:
                        raise ValueError("event_after_terminal")
                    if event == "[DONE]":
                        terminal = True
                        result["completion_ms"] = now_ms
                        result["status"] = "success"
                        continue
                    chunk = json.loads(event)
                    if chunk.get("model") != record["model"]:
                        raise ValueError("model_mismatch")
                    if result["first_event_ms"] is None:
                        result["first_event_ms"] = now_ms
                    choices = chunk.get("choices") or []
                    if choices and choices[0].get("delta", {}).get("content") and result["first_text_ms"] is None:
                        result["first_text_ms"] = now_ms
                    usage = chunk.get("usage")
                    if usage:
                        result.update({key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")})
                if not terminal:
                    raise ValueError("missing_terminal_marker")
    except asyncio.TimeoutError:
        result["status"], result["error_code"] = "timeout", "timeout"
    except asyncio.CancelledError:
        result["status"], result["error_code"] = "canceled", "canceled"
    except (httpx.HTTPError, OSError):
        result["status"], result["error_code"] = "error", "transport_error"
    except (ValueError, json.JSONDecodeError) as error:
        result["status"], result["error_code"] = "error", str(error) or "malformed_response"
    finally:
        result.setdefault("completion_ms", None)
    return result


async def replay_trace(records, endpoint, *, auth_token=None, max_inflight=64, max_connections=64, time_scale=1.0, drain_timeout_s=180.0):
    if time_scale <= 0 or max_inflight < 1 or max_connections < 1:
        raise ValueError("time_scale, max_inflight, and max_connections must be positive.")
    headers = {"authorization": f"Bearer {auth_token}"} if auth_token else {}
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    origin = perf_counter()
    active = set()
    tasks = []
    results = []

    async def run_one(client, record):
        task = asyncio.create_task(_request(client, endpoint, record, origin))
        cancel_after = record.get("cancel_after_s")
        if cancel_after is not None:
            done, _ = await asyncio.wait({task}, timeout=cancel_after)
            if not done:
                task.cancel()
        return await task

    async with httpx.AsyncClient(headers=headers, timeout=None, limits=limits, trust_env=False) as client:
        for record in records:
            target = record["scheduled_offset_s"] / time_scale
            await asyncio.sleep(max(0.0, target - (perf_counter() - origin)))
            if len(active) >= max_inflight:
                results.append({"result_schema_version": 1, "request_id": record["request_id"], "trace_id": record.get("trace_id"), "scenario_id": record["scenario_id"], "phase_id": record["phase_id"], "model": record["model"], "scheduled_offset_s": record["scheduled_offset_s"], "actual_submit_offset_s": None, "scheduling_delay_ms": None, "status": "not_submitted", "error_code": "client_inflight_limit"})
                continue
            task = asyncio.create_task(run_one(client, record))
            tasks.append(task)
            active.add(task)
            task.add_done_callback(active.discard)
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=drain_timeout_s)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            results.extend(task.result() for task in done if not task.cancelled() and task.exception() is None)
    return results


def write_results(results, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
