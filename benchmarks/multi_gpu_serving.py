"""Run the matched eight-case protocol against an approved static A-D serving deployment."""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import httpx

from benchmarks.client import _load_workload, _summary, run_request
from benchmarks.common import environment, write_json
from src.serving_config import load_config
from src.vllm_engine import HIGH_MODEL, LOW_MODEL

SHORT = "short-input-short-output"
LONG = "long-input-long-output"
REPEATS = 5
WARMUPS = 2
BURST = 8
MIXED_PER_MODEL = 4
LIFECYCLE_POLL_TIMEOUT = 30.0


def _lifecycle(before_metrics, gateway_stopped):
    startup = {
        "gateway_uptime_seconds": before_metrics.get("uptime_seconds"),
        "worker_ready_seconds_after_start": {
            name: row.get("ready_seconds_after_start")
            for name, row in before_metrics.get("workers", {}).items()},
    }
    worker_ready = startup["worker_ready_seconds_after_start"]
    return {
        "startup": startup,
        "cleanup": {"gateway_stopped": gateway_stopped},
        "complete": (
            isinstance(startup["gateway_uptime_seconds"], (int, float))
            and all(isinstance(value, (int, float)) for value in worker_ready.values())
            and gateway_stopped
        ),
    }


def _partition_by_model(rows, failed_model):
    affected = [row for row in rows if row["model"] == failed_model]
    unaffected = [row for row in rows if row["model"] != failed_model]
    return affected, unaffected


async def run_protocol(url, placement, config, small, large, failure_worker, output):
    parsed_url = httpx.URL(url)
    if (parsed_url.scheme not in {"http", "https"} or not parsed_url.host
            or parsed_url.username or parsed_url.password or parsed_url.path not in ("", "/")
            or parsed_url.query or parsed_url.fragment):
        raise ValueError("Server URL must be a credential-free HTTP(S) origin.")
    if placement != config.placement:
        raise ValueError("Selected placement does not match the serving config.")
    workloads = {LOW_MODEL: _load_workload(small), HIGH_MODEL: _load_workload(large)}
    for model, workload in workloads.items():
        if workload["model"] != model:
            raise ValueError("Workload file does not match its selected model.")
        expected = next(row.revision for row in config.models.values() if row.id == model)
        if workload["model_revision"] != expected or workload["tokenizer_revision"] != expected:
            raise ValueError("Workload model and tokenizer revisions must match the serving config.")
    expected_failure = {"A": "large-1", "B": "small-1", "C": "large-1", "D": "small-0"}[placement]
    if failure_worker != expected_failure:
        raise ValueError(f"Case 8 requires failure of configured worker {expected_failure} for placement {placement}.")
    failed_config = next(worker for worker in config.workers if worker.id == failure_worker)
    other_model = HIGH_MODEL if failed_config.model == LOW_MODEL else LOW_MODEL
    artifact = {
        "schema_version": 1, "experiment": "static-multi-gpu-serving",
        "status": "running", "placement": placement,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {"environment": environment(), "url": url,
                    "models": {model: {"revision": workload["model_revision"],
                                       "tokenizer_revision": workload["tokenizer_revision"],
                                       "chat_template_sha256": workload["chat_template_sha256"]}
                               for model, workload in workloads.items()}},
        "schedule": {"warmups_per_short_long_case": WARMUPS, "measured_repeats": REPEATS,
                     "same_model_burst": BURST, "mixed_requests_per_model": MIXED_PER_MODEL,
                     "sampling": {"temperature": 0, "top_p": 1},
                     "case_8_worker": failure_worker},
        "cases": [], "checks": {},
    }
    auth = os.environ.get("INFERENCE_API_TOKEN")
    headers = {"authorization": f"Bearer {auth}"} if auth else {}
    async with httpx.AsyncClient(timeout=180, headers=headers, trust_env=False,
                                 limits=httpx.Limits(max_connections=BURST, max_keepalive_connections=BURST)) as client:
        before = await client.get(url.rstrip("/") + "/ready")
        before_metrics = (await client.get(url.rstrip("/") + "/metrics")).json()
        if (before.status_code != 200
                or before_metrics.get("workers", {}).get(failure_worker, {}).get("state") != "ready"):
            raise RuntimeError("The selected failure worker must be ready before the benchmark starts.")
        artifact["case_8_worker_ready_before_run"] = True
        async def execute(name, targets, *, warmups=0, concurrent=False):
            rows = []
            started = None
            for phase, count in (("warmup", warmups), ("measured", REPEATS if warmups else 1)):
                if phase == "measured":
                    started = perf_counter()
                for iteration in range(1, count + 1):
                    calls = [run_request(client, url, workloads[model], case, phase, iteration)
                             for model, case in targets]
                    if concurrent:
                        rows.extend(await asyncio.gather(*calls))
                    else:
                        for call in calls:
                            rows.append(await call)
            elapsed = perf_counter() - started if started is not None else 0
            entry = {"name": name, "rows": rows, "elapsed_seconds": elapsed}
            entry["summary"] = _summary(rows, elapsed)
            artifact["cases"].append(entry)
            return entry

        singles = {}
        for model in (LOW_MODEL, HIGH_MODEL):
            workload = workloads[model]
            selected = {case["name"]: case for case in workload["cases"]}
            for label, case_name in (("short", SHORT), ("long", LONG)):
                case = {**selected[case_name], "model": model,
                        "model_revision": workload["model_revision"]}
                entry = await execute(f"{model}: {label} prompt/output", [(model, case)], warmups=WARMUPS)
                singles[(model, label)] = (workload, case)
                artifact["checks"][f"single_{'small' if model == LOW_MODEL else 'large'}_{label}"] = entry["summary"]["successes"] == REPEATS

        for model, label in ((LOW_MODEL, "small"), (HIGH_MODEL, "large")):
            workload, case = singles[(model, "short")]
            targets = [(model, case)] * BURST
            entry = await execute(f"{label}: simultaneous same-model burst", targets, concurrent=True)
            artifact["checks"][f"burst_{label}_bounded"] = (entry["summary"]["successes"] + entry["summary"]["rejections"] == BURST)

        mixed = [*([(LOW_MODEL, singles[(LOW_MODEL, "short")][1])] * MIXED_PER_MODEL),
                 *([(HIGH_MODEL, singles[(HIGH_MODEL, "short")][1])] * MIXED_PER_MODEL)]
        mixed_entry = await execute("mixed small+large concurrent traffic", mixed, concurrent=True)
        artifact["checks"]["mixed_model_identity"] = (
            len(mixed_entry["rows"]) == 2 * MIXED_PER_MODEL
            and all(row["status"] == "success" for row in mixed_entry["rows"])
        )

        print(f"Case 8: stop worker {failure_worker} at {failed_config.endpoint}; do not stop the gateway.")
        await asyncio.to_thread(input, "Wait until /ready reports it unready, then press Enter: ")
        ready = await client.get(url.rstrip("/") + "/ready")
        initial = ready.json()
        initial_metrics = (await client.get(url.rstrip("/") + "/metrics")).json()
        artifact["case_8_before"] = initial
        if ready.status_code == 200 and initial_metrics.get("workers", {}).get(failure_worker, {}).get("state") == "ready":
            raise RuntimeError("Selected failure worker is still ready; case 8 was not run.")
        failed_ready = initial_metrics.get("models", {}).get(failed_config.model, {}).get("ready", 0)
        should_retain = placement in {"B", "C"}
        artifact["case_8_expectation"] = "same-model replica remains routable" if should_retain else "model returns 503; no substitution to the idle/dedicated other GPU"
        failure_targets = [(failed_config.model, singles[(failed_config.model, "short")][1]),
                          (other_model, singles[(other_model, "short")][1])]
        failure_entry = await execute("replica/failure isolation", failure_targets, concurrent=True)
        affected, unaffected = _partition_by_model(failure_entry["rows"], failed_config.model)
        artifact["checks"]["case_8_worker_unready"] = initial_metrics.get("workers", {}).get(failure_worker, {}).get("state") in {"suspect", "unavailable"}
        artifact["checks"]["case_8_expected_availability"] = (
            failed_ready > 0 and bool(affected) and all(row["status"] == "success" for row in affected)
            if should_retain else failed_ready == 0 and bool(affected) and all(row["http_status"] == 503 for row in affected)
        )
        artifact["checks"]["case_8_other_model_unaffected"] = bool(unaffected) and all(row["status"] == "success" for row in unaffected)
        artifact["metrics_after"] = (await client.get(url.rstrip("/") + "/metrics")).json()

        print("Lifecycle: stop the gateway with its configured stop signal now.")
        await asyncio.to_thread(input, "Press Enter once the gateway process has exited: ")
        cleanup_started = perf_counter()
        gateway_stopped = False
        while perf_counter() - cleanup_started < LIFECYCLE_POLL_TIMEOUT:
            try:
                await client.get(url.rstrip("/") + "/health")
            except httpx.HTTPError:
                gateway_stopped = True
                break
            await asyncio.sleep(0.1)
        artifact["lifecycle"] = _lifecycle(before_metrics, gateway_stopped)
        artifact["checks"]["lifecycle_evidence"] = artifact["lifecycle"]["complete"]

    artifact["status"] = "completed" if all(artifact["checks"].values()) else "failed"
    write_json(output, artifact)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--small-workload", required=True)
    parser.add_argument("--large-workload", required=True)
    parser.add_argument("--failure-worker", required=True,
                        help="Worker to stop interactively before case 8; this client never kills it.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("Case 8 requires an interactive terminal for manual worker fault injection.")
    config = load_config(args.config)
    if Path(args.output).exists():
        parser.error("Output already exists; select a new artifact path.")
    try:
        result = asyncio.run(run_protocol(args.url, config.placement, config,
            args.small_workload, args.large_workload, args.failure_worker, args.output))
    except (OSError, ValueError, httpx.HTTPError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "placement": result["placement"],
                      "checks": result["checks"], "output": args.output}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
