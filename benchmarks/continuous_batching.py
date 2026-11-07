"""One bounded serial/refill observation against an existing vLLM HTTP server."""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from time import perf_counter


def new_result():
    """Create a JSON-ready record without importing an engine or loading weights."""
    return {
        "schema_version": 1,
        "kind": "continuous_batching",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_started",
        "runtime": {},
        "workload": None,
        "scenarios": [],
        "checks": {},
        "error": None,
        "measurement_scope": (
            "One observation per scenario on the same engine; no percentiles or "
            "comparison to previous CPU/T4 runs. Stream chunk gaps are not token latency. "
            "Client request overlap is not an engine scheduler trace."
        ),
    }


def _prompt(tokenizer, text):
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    token_ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
    return {
        "messages": messages,
        "formatted_prompt": formatted,
        "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
        "prompt_token_ids": token_ids,
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_tokens": len(token_ids),
    }


def _refill_checks(scenario):
    rows = {row["case"]: row for row in scenario["rows"]}
    checks = {
        "initial_request_overlap": False,
        "initial_text_overlap": False,
        "b_active_at_c_admission": False,
        "c_started_after_a_done": False,
        "c_first_text_before_b_done": False,
        "max_in_flight_at_most_two": scenario["max_in_flight"] <= 2,
    }
    if all(name in rows and rows[name]["status"] == "success" for name in ("A", "B", "C")):
        a, b, c = (rows[name]["timeline"] for name in ("A", "B", "C"))
        checks["initial_request_overlap"] = (
            max(a["request_start_ms"], b["request_start_ms"]) < min(a["done_ms"], b["done_ms"])
        )
        checks["initial_text_overlap"] = (
            a["first_text_ms"] is not None and b["first_text_ms"] is not None
            and max(a["first_text_ms"], b["first_text_ms"]) < min(a["done_ms"], b["done_ms"])
        )
        checks["b_active_at_c_admission"] = c["admitted_ms"] < b["done_ms"]
        checks["c_started_after_a_done"] = c["request_start_ms"] >= a["done_ms"]
        checks["c_first_text_before_b_done"] = (
            c["first_text_ms"] is not None and c["first_text_ms"] < b["done_ms"]
        )
    # Text overlap is also required: queued HTTP requests alone do not show active streams.
    checks["refill_demonstrated"] = all(checks.values())
    return checks


async def _scenario(
    client, url, workload, cases, scenario, *, refill=False, fixed=True, vllm_options=True,
):
    from benchmarks.client import run_request

    started = perf_counter()
    tasks = []
    active = 0
    scenario.update(status="running", rows=[], max_in_flight=0, checks={})

    async def request(case, row):
        nonlocal active
        active += 1
        scenario["max_in_flight"] = max(scenario["max_in_flight"], active)
        try:
            options = {}
            if vllm_options:
                options.update(
                    stream_options={"include_usage": True},
                    return_token_ids=True,
                    stop_token_ids=workload["eos_token_ids"],
                )
                if fixed:
                    options["min_tokens"] = case["max_tokens"]
            result = await run_request(
                client, url, workload, case, scenario["name"], 1,
                request_options=options, capture_events=True,
            )
            row.update(result)
            actual_start = (row["started_at"] - started) * 1000
            row["timeline"].update(
                request_start_ms=actual_start,
                first_text_ms=(actual_start + row["first_text_ms"]
                               if row["first_text_ms"] is not None else None),
                done_ms=(actual_start + row["e2e_ms"] if row["status"] == "success" else None),
            )
            events = row["events"]
            row["stream_chunk_gaps_ms"] = [
                following["elapsed_ms"] - previous["elapsed_ms"]
                for previous, following in zip(events, events[1:])
            ]
            row["output_token_ids"] = [
                token for event in events for token in (event["token_ids"] or [])
            ]
            returned_prompt = row.get("prompt_token_ids")
            row["checks"] = {
                "valid_usage_and_terminal_stream": row["status"] == "success",
                "fixed_completion_count_matches_cap": (
                    row["completion_tokens"] == case["max_tokens"] if fixed else None
                ),
                "prompt_token_ids_match": (
                    returned_prompt == case["prompt_token_ids"] if returned_prompt is not None else None
                ),
            }
            if row["status"] != "success":
                if row["error_code"] == "interrupted":
                    raise asyncio.CancelledError
                raise RuntimeError(f"{scenario['name']}/{case['name']}: {row['error_code']}")
            if any(value is False for value in row["checks"].values()):
                raise RuntimeError(f"{scenario['name']}/{case['name']}: request checks failed")
        except BaseException as error:
            row["measurement_error"] = f"{type(error).__name__}: {error}"
            if row["status"] == "running":
                row["status"] = "canceled" if isinstance(error, asyncio.CancelledError) else "error"
            raise
        finally:
            row["timeline"]["ended_ms"] = (perf_counter() - started) * 1000
            active -= 1
        return row

    def submit(case, planned_start, after):
        row = {
            "case": case["name"], "status": "running", "max_tokens": case["max_tokens"],
            "planned_after": after,
            "timeline": {
                "planned_start_ms": planned_start,
                "admitted_ms": (perf_counter() - started) * 1000,
                "request_start_ms": None, "first_text_ms": None,
                "done_ms": None, "ended_ms": None,
            },
        }
        scenario["rows"].append(row)
        task = asyncio.create_task(request(case, row))
        tasks.append(task)
        return task

    try:
        if refill:
            a = submit(cases[0], 0.0, "scenario_start")
            b = submit(cases[1], 0.0, "scenario_start")
            pending = {a, b}
            while not a.done():
                completed, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in completed:
                    task.result()  # A failure in either initial request stops admission immediately.
            a_row = a.result()
            if b.done():
                b.result()
            scenario["b_http_task_pending_at_c_admission"] = not b.done()
            c = submit(cases[2], a_row["timeline"]["done_ms"], "A.done")
            await asyncio.gather(b, c)
        else:
            previous = None
            for case in cases:
                previous = await submit(
                    case, previous["timeline"]["done_ms"] if previous else 0.0,
                    previous["case"] + ".done" if previous else "scenario_start",
                )
        scenario["status"] = "completed"
    except BaseException:
        scenario["status"] = "failed"
        raise
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for row in scenario["rows"]:
            if row["status"] == "running":
                row.update(status="canceled", error_code="not_started_before_cancellation")
        scenario["elapsed_ms"] = (perf_counter() - started) * 1000
        successes = [row for row in scenario["rows"] if row["status"] == "success"]
        tokens = sum(row["completion_tokens"] for row in successes)
        scenario["aggregate"] = {
            "successful_requests": len(successes),
            "successful_prompt_tokens": sum(row["prompt_tokens"] for row in successes),
            "successful_requests_per_second": len(successes) * 1000 / scenario["elapsed_ms"],
            "successful_completion_tokens": tokens,
            "output_tokens_per_second": tokens * 1000 / scenario["elapsed_ms"],
            "throughput_includes_only_successful_usage": True,
        }
        if refill:
            scenario["checks"].update(_refill_checks(scenario))
        elif not fixed and successes:
            row = successes[0]
            scenario["checks"] = {
                "valid_usage_and_terminal_stream": True,
                "early_eos": row["finish_reason"] == "stop" and 0 < row["completion_tokens"] < cases[0]["max_tokens"],
                "last_returned_token_is_eos": (
                    row["output_token_ids"][-1] in workload["eos_token_ids"]
                    if row["output_token_ids"] else None
                ),
            }


async def measure(url, record, tokenizer, eos_token_ids):
    """Mutate and return record; retain partial rows and cancel HTTP work on failure."""
    import httpx

    from benchmarks.workloads import MODEL_ID, SHORT_PROMPT

    record["status"] = "running"
    try:
        if not eos_token_ids or any(type(token) is not int or token < 0 for token in eos_token_ids):
            raise ValueError("Pinned generation config must supply nonnegative integer EOS IDs.")
        prompt = _prompt(tokenizer, SHORT_PROMPT)
        cases = [{**prompt, "name": name, "max_tokens": cap} for name, cap in (("A", 16), ("B", 128), ("C", 32))]
        natural = {
            **_prompt(tokenizer, "Reply with exactly one word: Hello"),
            "name": "natural_eos", "max_tokens": 32,
        }
        for case in [*cases, natural]:
            if case["prompt_tokens"] + case["max_tokens"] > 2048:
                raise ValueError(f"{case['name']} exceeds the 2048-token context budget.")
        workload = {
            "model": MODEL_ID,
            "generation": {"temperature": 0, "top_p": 1},
            "context_limit": 2048,
            "eos_token_ids": list(eos_token_ids),
            "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
            "cases": cases,
            "natural_eos_case": natural,
        }
        record["workload"] = workload
        limits = httpx.Limits(max_connections=2, max_keepalive_connections=2)
        async with httpx.AsyncClient(timeout=120, limits=limits, trust_env=False) as client:
            for name, selected in (
                ("warmup", [cases[0]]), ("sequential", cases),
                ("refill", cases), ("natural_eos", [natural]),
            ):
                scenario = {
                    "name": name,
                    "discarded": name == "warmup",
                    "admission_plan": (
                        "A and B at scenario start; C immediately after A finishes"
                        if name == "refill" else "One request at a time, in listed order"
                    ),
                    "planned_cases": [case["name"] for case in selected],
                }
                record["scenarios"].append(scenario)
                await _scenario(
                    client, url, workload, selected, scenario,
                    refill=name == "refill", fixed=name != "natural_eos",
                )
        scenarios = {scenario["name"]: scenario for scenario in record["scenarios"]}
        record["checks"] = {
            "requests_valid": True,
            "refill_demonstrated": scenarios["refill"]["checks"]["refill_demonstrated"],
            "natural_eos_demonstrated": scenarios["natural_eos"]["checks"]["early_eos"],
        }
        record["checks"]["all_passed"] = all(record["checks"].values())
        record["status"] = "completed"
    except BaseException as error:
        record["status"] = "failed"
        record["checks"]["all_passed"] = False
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    return record
