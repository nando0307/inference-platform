"""Raw result persistence and load-generator accuracy summaries."""

import json
from pathlib import Path


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, round(fraction * (len(values) - 1)))
    return values[index]


def summarize_results(results, intended_count=None):
    scheduled = intended_count if intended_count is not None else len(results)
    submitted = sum(row.get("actual_submit_offset_s") is not None for row in results)
    not_submitted = sum(row.get("status") == "not_submitted" for row in results)
    delays = [row["scheduling_delay_ms"] for row in results if row.get("scheduling_delay_ms") is not None]
    submitted_ratio = submitted / scheduled if scheduled else 1.0
    return {
        "scheduled_requests": scheduled,
        "submitted_requests": submitted,
        "not_submitted_requests": not_submitted,
        "submission_ratio": submitted_ratio,
        "scheduling_delay_ms": {
            "count": len(delays),
            "p50": percentile(delays, 0.50),
            "p95": percentile(delays, 0.95),
            "p99": percentile(delays, 0.99),
        },
        "limited_by_client": not_submitted > 0 or submitted_ratio < 0.99,
    }


def write_results(results, path: str | Path, *, intended_count=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    path.with_suffix(".summary.json").write_text(
        json.dumps(summarize_results(results, intended_count=intended_count), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
