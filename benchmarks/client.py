"""Sequential streaming HTTP measurements; this module never loads model weights."""

import argparse
import json
import math
import re
from pathlib import Path
from time import perf_counter

import httpx

from benchmarks.common import distribution, environment, write_json


class StreamError(ValueError):
    pass


def _events(response):
    """Dispatch complete SSE data events, including multiline data and CRLF."""
    data = []
    for line in response.iter_lines():
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif not line.startswith(":"):
            field, _, value = line.partition(":")
            if field == "data":
                data.append(value.removeprefix(" "))
    # An unterminated event is not a received [DONE] marker.
    if data:
        raise StreamError("truncated_stream")


def _error_code(payload, default):
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", code):
        return code
    return default


def _terminal_usage(usage, case, finish_reason):
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(usage, dict) or any(
        type(usage.get(field)) is not int or usage[field] < 0 for field in fields
    ):
        raise StreamError("invalid_usage")
    counts = {field: usage[field] for field in fields}
    if (
        counts["prompt_tokens"] != case["prompt_tokens"]
        or counts["completion_tokens"] > case["max_tokens"]
        or counts["total_tokens"] != counts["prompt_tokens"] + counts["completion_tokens"]
        or (finish_reason == "length" and counts["completion_tokens"] != case["max_tokens"])
    ):
        raise StreamError("invalid_usage")
    return counts


def _consume_stream(response, row, case, started):
    finish_reason = None
    usage = None
    done = False
    for event in _events(response):
        if done:
            raise StreamError("malformed_stream")
        if event == "[DONE]":
            if finish_reason is None:
                raise StreamError("missing_terminal")
            if usage is None:
                raise StreamError("missing_usage")
            if row["first_text_ms"] is not None and usage["completion_tokens"] == 0:
                raise StreamError("invalid_usage")
            row["e2e_ms"] = (perf_counter() - started) * 1000
            done = True
            continue
        try:
            chunk = json.loads(event)
        except ValueError:
            raise StreamError("malformed_stream") from None
        if not isinstance(chunk, dict):
            raise StreamError("malformed_stream")
        if "error" in chunk:
            raise StreamError(_error_code(chunk, "stream_error"))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamError("malformed_stream")
        if choices:
            choice = choices[0]
            if (
                finish_reason is not None
                or not isinstance(choice, dict)
                or type(choice.get("index")) is not int
                or choice["index"] != 0
                or not isinstance(choice.get("delta"), dict)
            ):
                raise StreamError("malformed_stream")
            delta = choice["delta"]
            if delta.get("role", "assistant") != "assistant":
                raise StreamError("malformed_stream")
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise StreamError("malformed_stream")
            if content and row["first_text_ms"] is None:
                row["first_text_ms"] = (perf_counter() - started) * 1000
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None and finish_reason not in ("stop", "length"):
                raise StreamError("invalid_finish_reason")
        elif finish_reason is None or chunk.get("usage") is None:
            raise StreamError("malformed_stream")
        if chunk.get("usage") is not None:
            if finish_reason is None or usage is not None:
                raise StreamError("invalid_usage")
            usage = _terminal_usage(chunk["usage"], case, finish_reason)
    if not done:
        raise StreamError("truncated_stream")
    row.update(usage, finish_reason=finish_reason, status="success")


def run_request(client, url, workload, case, phase, iteration):
    row = {
        "case": case["name"], "phase": phase, "iteration": iteration,
        "status": "error", "http_status": None, "request_id": None,
        "e2e_ms": None, "first_text_ms": None, "prompt_tokens": None,
        "completion_tokens": None, "total_tokens": None, "finish_reason": None,
        "error_code": None,
    }
    payload = {
        "model": workload["model"], "messages": case["messages"],
        "max_tokens": case["max_tokens"], "temperature": workload["generation"]["temperature"],
        "top_p": workload["generation"]["top_p"], "stream": True,
    }
    started = perf_counter()
    try:
        with client.stream("POST", url.rstrip("/") + "/v1/chat/completions", json=payload) as response:
            row["http_status"] = response.status_code
            row["request_id"] = response.headers.get("x-request-id")
            if response.status_code != 200:
                response.read()
                try:
                    error = response.json()
                except ValueError:
                    error = None
                row["error_code"] = _error_code(error, "http_error")
            elif response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                row["error_code"] = "unexpected_content_type"
            else:
                response.encoding = "utf-8"
                _consume_stream(response, row, case, started)
    except StreamError as error:
        row["error_code"] = str(error)
    except httpx.TimeoutException:
        row["error_code"] = "timeout"
    except httpx.HTTPError:
        row["error_code"] = "transport_error"
    except KeyboardInterrupt:
        row["status"] = "canceled"
        row["error_code"] = "interrupted"
    finally:
        if row["status"] != "success" or row["error_code"] is not None or row["e2e_ms"] is None:
            row["e2e_ms"] = (perf_counter() - started) * 1000
    if row["error_code"] is not None and row["status"] != "canceled":
        row["status"] = "error"
    return row


def _summary(rows, elapsed):
    measured = [row for row in rows if row["phase"] == "measured"]
    successes = [row for row in measured if row["status"] == "success"]
    tokens = sum(row["completion_tokens"] for row in successes)
    return {
        "measured_requests": len(measured), "successes": len(successes),
        "errors": sum(row["status"] == "error" for row in measured),
        "canceled": sum(row["status"] == "canceled" for row in measured),
        "warmup_errors": sum(row["phase"] == "warmup" and row["status"] != "success" for row in rows),
        "successful_completion_tokens": tokens,
        "output_tokens_per_second": tokens / elapsed if elapsed > 0 else None,
        "e2e_ms": distribution([row["e2e_ms"] for row in successes]),
        "first_text_ms": distribution([row["first_text_ms"] for row in successes]),
    }


def run_benchmark(url, workload, cases, *, server_config=None, warmups=2, repeats=20, timeout=120):
    result = {
        "schema_version": 1, "kind": "http", "environment": environment(),
        "workload": workload, "declared_server_config": server_config,
        "url": url, "timeout_seconds": timeout, "warmups": warmups, "repeats": repeats,
        "interrupted": False, "cases": [], "samples": [],
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            for case in cases:
                entry = {"name": case["name"], "elapsed_seconds": 0.0, "summary": {}}
                result["cases"].append(entry)
                rows = []
                measured_start = None
                try:
                    for phase, count in (("warmup", warmups), ("measured", repeats)):
                        if phase == "measured":
                            measured_start = perf_counter()
                        for iteration in range(1, count + 1):
                            row = run_request(client, url, workload, case, phase, iteration)
                            rows.append(row)
                            result["samples"].append(row)
                            if row["status"] == "canceled":
                                raise KeyboardInterrupt
                finally:
                    if measured_start is not None:
                        entry["elapsed_seconds"] = perf_counter() - measured_start
                    entry["summary"] = _summary(rows, entry["elapsed_seconds"])
    except KeyboardInterrupt:
        result["interrupted"] = True
    return result


def _load_workload(path):
    with Path(path).open(encoding="utf-8") as handle:
        workload = json.load(handle)
    if not isinstance(workload, dict):
        raise ValueError("Workload must be a JSON object")
    baseline = workload.get("baseline", {})
    generation = workload.get("generation", {})
    if (
        workload.get("schema_version") != 1
        or workload.get("model") != "Qwen/Qwen2.5-0.5B-Instruct"
        or not isinstance(workload.get("model_revision"), str) or not workload["model_revision"]
        or not isinstance(workload.get("tokenizer_revision"), str) or not workload["tokenizer_revision"]
        or not isinstance(baseline, dict)
        or baseline.get("device") != "cpu"
        or baseline.get("dtype") != "float32"
        or baseline.get("workers") != 1
        or baseline.get("concurrency") != 1
        or baseline.get("context_limit") != 2048
        or baseline.get("eos_counted") is not True
        or not isinstance(generation, dict)
        or type(generation.get("temperature")) not in (int, float)
        or generation["temperature"] != 0
        or type(generation.get("top_p")) not in (int, float)
        or generation["top_p"] != 1
    ):
        raise ValueError("Workload must declare the versioned Qwen greedy baseline and EOS-inclusive counts")
    cases = workload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Workload needs at least one case")
    names = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("name"), str) or not case["name"]:
            raise ValueError("Every workload case needs a name")
        if case["name"] in names:
            raise ValueError("Workload case names must be unique")
        names.add(case["name"])
        if (
            type(case.get("prompt_tokens")) is not int or case["prompt_tokens"] < 1
            or type(case.get("max_tokens")) is not int or not 1 <= case["max_tokens"] <= 512
            or case["prompt_tokens"] + case["max_tokens"] > baseline["context_limit"]
        ):
            raise ValueError("Workload token counts must fit the declared context and output limits")
        messages = case.get("messages")
        if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) or set(message) != {"role", "content"}
            or message["role"] not in ("user", "assistant", "system")
            or not isinstance(message["content"], str) or not message["content"].strip()
            for message in messages
        ):
            raise ValueError("Workload messages must use the supported text chat contract")
    return workload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Server base URL, for example http://127.0.0.1:8000")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--server-config", help="JSON object of declared, not detected, server configuration")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--case", action="append", dest="case_names")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Require warmups >= 0, repeats >= 1, and a finite positive timeout")
    try:
        url = httpx.URL(args.url)
        if url.scheme not in ("http", "https") or not url.host or url.query or url.fragment:
            raise ValueError("URL must be an HTTP(S) base URL without a query or fragment")
        if Path(args.output).exists():
            raise ValueError("Output already exists; select a new evidence file")
        workload = _load_workload(args.workload)
        server_config = None
        if args.server_config:
            with Path(args.server_config).open(encoding="utf-8") as handle:
                server_config = json.load(handle)
            if not isinstance(server_config, dict):
                raise ValueError("Server configuration must be a JSON object")
        cases = workload["cases"]
        if args.case_names:
            names = set(args.case_names)
            if names - {case["name"] for case in cases}:
                raise ValueError("Unknown workload case")
            cases = [case for case in cases if case["name"] in names]
    except (OSError, ValueError, httpx.InvalidURL) as error:
        parser.error(str(error))
    result = run_benchmark(
        args.url, workload, cases, server_config=server_config,
        warmups=args.warmups, repeats=args.repeats, timeout=args.timeout,
    )
    write_json(args.output, result)
    for case in result["cases"]:
        summary = case["summary"]
        latency = summary["e2e_ms"]
        first = summary["first_text_ms"]["p50"]
        throughput = summary["output_tokens_per_second"]
        def number(value):
            return "n/a" if value is None else f"{value:.2f}"
        print(
            f"{case['name']}: success={summary['successes']}/{summary['measured_requests']} "
            f"errors={summary['errors']} canceled={summary['canceled']} "
            f"warmup_failures={summary['warmup_errors']} "
            f"E2E p50/p95={number(latency['p50'])}/{number(latency['p95'])} ms "
            f"first-text p50={number(first)} ms tokens/s={number(throughput)}"
        )
    print("Latency percentiles use successful measured samples; small-sample p95 is exploratory.")
    print(f"Saved {args.output}")
    if result["interrupted"]:
        return 130
    return int(any(row["status"] != "success" for row in result["samples"]))


if __name__ == "__main__":
    raise SystemExit(main())
