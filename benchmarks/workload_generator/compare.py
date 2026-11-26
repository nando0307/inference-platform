"""Compare two replays of the same pinned trace without hiding failed requests."""

import argparse
import json
from collections import Counter
from pathlib import Path

from benchmarks.common import distribution, write_json
from .trace import read_trace

METRICS = ("completion_ms", "first_text_ms", "scheduling_delay_ms", "prompt_tokens", "completion_tokens")
STATUSES = ("success", "error", "rejected", "canceled", "timeout", "not_submitted")


def _index(rows, trace, label):
    indexed = {}
    for row in rows:
        request_id = row.get("request_id")
        if request_id not in trace or request_id in indexed:
            raise ValueError(f"{label}: unknown or duplicate trace request ID: {request_id}")
        expected = trace[request_id]
        for key in ("model", "model_revision", "trace_id", "scenario_id", "phase_id", "scheduled_offset_s"):
            if row.get(key) is not None and row[key] != expected.get(key):
                raise ValueError(f"{label}: {key} mismatch for {request_id}")
        if row.get("status") not in STATUSES:
            raise ValueError(f"{label}: unknown status for {request_id}")
        indexed[request_id] = row
    return indexed


def compare(trace_rows, meta, baseline_rows, candidate_rows):
    """Return deterministic, per-request and per-phase/model comparisons."""
    if meta.get("request_count") != len(trace_rows):
        raise ValueError("Trace metadata request_count does not match the trace.")
    trace = {}
    for row in trace_rows:
        request_id, model = row["request_id"], row["model"]
        pinned = meta.get("models", {}).get(model, {}).get("revision")
        if not pinned or row.get("model_revision") != pinned:
            raise ValueError(f"Trace model revision mismatch for {request_id}.")
        if row.get("trace_id") != meta.get("trace_id") or request_id in trace:
            raise ValueError(f"Trace identity mismatch or duplicate request ID: {request_id}.")
        trace[request_id] = row
    baseline = _index(baseline_rows, trace, "baseline")
    candidate = _index(candidate_rows, trace, "candidate")
    samples = [{"request_id": row["request_id"], "phase_id": row["phase_id"], "model": row["model"],
                "baseline": baseline.get(row["request_id"]), "candidate": candidate.get(row["request_id"])}
               for row in trace_rows]
    groups = {("all", "all"): samples}
    for sample in samples:
        groups.setdefault((sample["phase_id"], sample["model"]), []).append(sample)
    warnings = []
    summaries = []
    for (phase, model), members in sorted(groups.items()):
        label = f"{phase}/{model}"
        sides = {}
        for side in ("baseline", "candidate"):
            rows = [sample[side] for sample in members]
            status = Counter(row["status"] for row in rows if row is not None)
            missing = len(rows) - sum(status.values())
            if missing:
                warnings.append(f"{label} {side}: {missing} missing result observations.")
            metrics = {}
            for metric in METRICS:
                eligible = [row for row, sample in zip(rows, members) if row is not None and (
                    row["status"] != "not_submitted" if metric == "scheduling_delay_ms" else
                    row["status"] == "success" and (metric != "first_text_ms" or trace[sample["request_id"]]["stream"]))]
                values = [row.get(metric) for row in eligible]
                metrics[metric] = distribution(values)
                absent = len(eligible) - metrics[metric]["n"]
                if absent:
                    warnings.append(f"{label} {side}: {absent} missing {metric} observations (of {len(eligible)} eligible).")
            sides[side] = {"scheduled": len(rows), "missing_results": missing,
                           "statuses": {key: status[key] for key in STATUSES}, "metrics": metrics}
        delta = {metric: {quantile: (sides["candidate"]["metrics"][metric][quantile] -
                                    sides["baseline"]["metrics"][metric][quantile]
                                    if sides["baseline"]["metrics"][metric][quantile] is not None and
                                    sides["candidate"]["metrics"][metric][quantile] is not None else None)
                          for quantile in ("p50", "p95")} for metric in METRICS}
        summaries.append({"phase_id": phase, "model": model, **sides, "candidate_minus_baseline": delta})
    if any(row.get("model_revision") is None for row in baseline_rows + candidate_rows):
        warnings.append("Replay result model revisions unavailable; older results cannot confirm the declared revision.")
    warnings.append("The endpoint's actual loaded revision is not exposed by replay; pinned trace revisions are declarations, not server verification.")
    return {"schema_version": 1, "kind": "workload_replay_comparison", "trace_id": meta["trace_id"],
            "models": meta["models"], "request_count": len(trace_rows), "groups": summaries,
            "samples": samples, "warnings": warnings}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    def read_jsonl(path):
        with Path(path).open(encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    trace_rows = read_trace(args.trace)
    meta = json.loads(Path(args.trace).with_suffix(".meta.json").read_text(encoding="utf-8"))
    report = compare(trace_rows, meta, read_jsonl(args.baseline), read_jsonl(args.candidate))
    write_json(args.output, report)
    print(json.dumps({"output": args.output, "requests": report["request_count"],
                      "warnings": report["warnings"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
