"""Join client delivery metrics to opt-in server token metrics by request ID."""

import argparse
import json
from pathlib import Path

from benchmarks.common import distribution, write_json


def build_report(client, server_records):
    startups = [row for row in server_records if row.get("type") == "startup"]
    if server_records and len(startups) != 1:
        raise ValueError("Server metrics must contain exactly one startup record.")
    startup = startups[0] if startups else None
    if startup:
        revision = startup["runtime"]["model_revision"]
        if revision != client["workload"]["model_revision"]:
            raise ValueError("Workload and measured server model revisions differ.")
    model_rows = {}
    scheduling_rows = {}
    for row in server_records:
        kind = row.get("type")
        if kind not in ("generation", "scheduling"):
            continue
        indexed = model_rows if kind == "generation" else scheduling_rows
        request_id = row["request_id"]
        if not request_id or request_id in indexed:
            raise ValueError(f"{kind.capitalize()} records need unique, nonempty request IDs.")
        indexed[request_id] = row

    cases = []
    joined_samples = []
    warnings = []
    for case in client["cases"]:
        samples = [row for row in client["samples"]
                   if row["case"] == case["name"] and row["phase"] == "measured"]
        successful = [row for row in samples if row["status"] == "success"]
        failed = [row for row in samples if row["status"] != "success"]
        measured_model = []
        for row in successful:
            model = model_rows.get(row["request_id"])
            if model is None:
                continue
            if model["status"] != "success":
                raise ValueError("Successful HTTP delivery has a failed model record.")
            if any(model[key] != row[key] for key in ("prompt_tokens", "completion_tokens")):
                raise ValueError("Client usage and model token counts disagree.")
            measured_model.append(model)
        failed_model = [model_rows[row["request_id"]] for row in failed
                        if row["request_id"] in model_rows]
        joined = []
        for row in samples:
            request_id = row["request_id"]
            model = model_rows.get(request_id)
            scheduling = scheduling_rows.get(request_id)
            queue_source = scheduling if scheduling is not None else model
            joined.append({
                **row,
                "model_status": model.get("status") if model is not None else None,
                "scheduling_state": scheduling.get("state") if scheduling is not None else None,
                "queue_wait_ms": queue_source.get("queue_wait_ms") if queue_source is not None else None,
                "cancellation_delay_ms": scheduling.get("cancellation_delay_ms") if scheduling is not None else None,
                "cancellation_reason": scheduling.get("cancellation_reason") if scheduling is not None else None,
            })
        joined_samples.extend(joined)
        missing_queue = sum(row["status"] == "success" and row["queue_wait_ms"] is None for row in joined)
        missing_cancellation = sum(
            (row["status"] == "canceled" and row["scheduling_state"] is None)
            or (row["scheduling_state"] == "canceled" and row["queue_wait_ms"] is not None
                and row["cancellation_delay_ms"] is None)
            for row in joined
        )
        if missing_queue:
            warnings.append(f"{case['name']}: queue wait unavailable for {missing_queue} successful requests.")
        if missing_cancellation:
            warnings.append(
                f"{case['name']}: cancellation completion measurements unavailable for "
                f"{missing_cancellation} requests; the producer may still be exiting."
            )
        elapsed = case["elapsed_seconds"]
        if elapsed < 0 or (samples and elapsed == 0):
            raise ValueError("Measured requests require a positive workload duration.")
        completed_tokens = sum(row["completion_tokens"] for row in successful)
        memory = [row["process_peak_rss_bytes"] for row in measured_model + failed_model]
        cases.append({
            "name": case["name"],
            "requests": len(samples),
            "successes": len(successful),
            "errors": sum(row["status"] == "error" and row.get("http_status") != 429 for row in failed),
            "rejections": sum(row.get("http_status") == 429 for row in samples),
            "canceled": sum(row["status"] == "canceled" for row in failed),
            "warmup_errors": sum(row["status"] != "success" for row in client["samples"]
                                 if row["case"] == case["name"] and row["phase"] == "warmup"),
            "elapsed_seconds": elapsed,
            "successful_completion_tokens": completed_tokens,
            "useful_output_tokens_per_second": completed_tokens / elapsed if elapsed > 0 else None,
            "prompt_tokens": distribution(row["prompt_tokens"] for row in successful),
            "completion_tokens": distribution(row["completion_tokens"] for row in successful),
            "e2e_ms": distribution(row["e2e_ms"] for row in successful),
            "client_first_text_ms": distribution(row["first_text_ms"] for row in successful),
            "queue_wait_ms": distribution(row["queue_wait_ms"] for row in joined),
            "missing_successful_queue_samples": missing_queue,
            "scheduling_samples": sum(row["request_id"] in scheduling_rows for row in samples),
            "server_canceled": sum(row["scheduling_state"] == "canceled" for row in joined),
            "canceled_before_start": sum(
                row["scheduling_state"] == "canceled" and row["queue_wait_ms"] is None
                for row in joined
            ),
            "cancellation_delay_ms": distribution(row["cancellation_delay_ms"] for row in joined),
            "missing_cancellation_delay_samples": missing_cancellation,
            "model_samples": len(measured_model),
            "missing_successful_model_samples": len(successful) - len(measured_model),
            "model_ttft_ms": distribution(row["ttft_ms"] for row in measured_model),
            "model_tpot_ms": distribution(row["tpot_ms"] for row in measured_model),
            "model_generation_ms": distribution(row["generation_ms"] for row in measured_model),
            "inter_token_ms": distribution(value for row in measured_model for value in row["inter_token_ms"]),
            "process_peak_rss_bytes": max(memory) if memory else None,
            "failed_or_canceled_model_samples": len(failed_model),
            "known_failed_or_canceled_generation_ms": (
                sum(row["generation_ms"] for row in failed_model) if failed_model else None
            ),
        })
    return {
        "schema_version": 1,
        "kind": "http_with_model_metrics",
        "client_environment": client["environment"],
        "declared_server_config": client["declared_server_config"],
        "server_startup": startup,
        "workload": client["workload"],
        "warmups": client["warmups"],
        "repeats": client["repeats"],
        "concurrency": client.get("concurrency", 1),
        "workload_mode": client.get("workload_mode", "sequential"),
        "cancel_after_text": client.get("cancel_after_text", False),
        "interrupted": client.get("interrupted", False),
        "cases": cases,
        "samples": joined_samples,
        "warnings": warnings,
        "limitations": [
            "Model and client durations use separate monotonic clocks; only request IDs are joined.",
            "Queue wait is validated admission to actual worker start, separate from model TTFT and client first text.",
            "Cancellation delay is signal to actual worker exit, not the client's response-close time.",
            "Queued cancellations never start generation and have no queue-wait or worker-cancellation-delay measurement.",
            "Terminal scheduling records are authoritative; metrics collected before worker exit may be missing.",
            "Warm-ups are excluded. Each case elapsed window includes failures and client overhead, but not other cases.",
            "Useful throughput counts only successfully delivered generated IDs, including EOS.",
            "Process peak RSS is a lifetime high-water mark, not per-case allocation or accelerator memory.",
            "Missing model or scheduling measurements, including older Phase 3 artifacts, remain unavailable, not zero.",
            "Small-sample p95 is exploratory, not a stable production tail-latency estimate.",
            "Token callbacks and metrics output add measurement overhead; profiling is run separately.",
        ],
    }


def print_table(report):
    def number(value):
        return "n/a" if value is None else f"{value:.2f}"

    print(f"Workload: {report['workload_mode']}, concurrency={report['concurrency']}, "
          f"total measured requests/case={report['repeats']}")
    print("case | requests/errors/rejected429/canceled/warmup-failures | E2E p50/p95 ms | first text p50 ms | "
          "queue wait p50/p95 ms | TTFT p50 ms | TPOT p50 ms | cancel delay p50 ms | "
          "useful output tok/s | peak RSS MiB | model samples")
    for case in report["cases"]:
        memory = case["process_peak_rss_bytes"]
        print(f"{case['name']} | {case['requests']}/{case['errors']}/{case['rejections']}/"
              f"{case['canceled']}/{case['warmup_errors']} | "
              f"{number(case['e2e_ms']['p50'])}/{number(case['e2e_ms']['p95'])} | "
              f"{number(case['client_first_text_ms']['p50'])} | "
              f"{number(case['queue_wait_ms']['p50'])}/{number(case['queue_wait_ms']['p95'])} | "
              f"{number(case['model_ttft_ms']['p50'])} | {number(case['model_tpot_ms']['p50'])} | "
              f"{number(case['cancellation_delay_ms']['p50'])} | "
              f"{number(case['useful_output_tokens_per_second'])} | "
              f"{number(memory / 2**20 if memory is not None else None)} | "
              f"{case['model_samples']}/{case['successes']}")
    for warning in report["warnings"]:
        print(f"Warning: {warning}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True)
    parser.add_argument("--server-metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    client = json.loads(Path(args.client).read_text(encoding="utf-8"))
    records = []
    if args.server_metrics:
        records = [json.loads(line) for line in Path(args.server_metrics).read_text(encoding="utf-8").splitlines() if line.strip()]
    report = build_report(client, records)
    write_json(args.output, report)
    print_table(report)


if __name__ == "__main__":
    main()
