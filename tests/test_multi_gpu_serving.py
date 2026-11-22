import unittest

from benchmarks.multi_gpu_serving import REPEATS, WARMUPS, _partition_by_model, _summary


def row(phase, *, status="success", http_status=200, tokens=10, e2e_ms=100.0, first_text_ms=20.0):
    return {"phase": phase, "status": status, "http_status": http_status,
            "completion_tokens": tokens, "e2e_ms": e2e_ms, "first_text_ms": first_text_ms}


def failure_row(model, *, status="success", http_status=200):
    return {"case": "short-input-short-output", "model": model,
            "status": status, "http_status": http_status}


class SummaryTests(unittest.TestCase):
    def test_summary_aggregates_only_measured_rows(self):
        rows = ([row("warmup", tokens=1000, e2e_ms=9000.0) for _ in range(WARMUPS)]
                + [row("measured") for _ in range(REPEATS)])
        summary = _summary(rows, 2.0)
        self.assertEqual(summary["requests"], REPEATS)
        self.assertEqual(summary["successes"], REPEATS)
        self.assertEqual(summary["successful_completion_tokens"], REPEATS * 10)
        self.assertEqual(summary["output_tokens_per_second"], REPEATS * 10 / 2.0)
        self.assertEqual(summary["e2e_ms"]["max"], 100.0)
        self.assertEqual(summary["first_text_ms"]["max"], 20.0)

    def test_summary_counts_measured_rejections_errors_and_cancellations(self):
        rows = [row("warmup")]
        rows += [row("measured", status="rejected", http_status=429)]
        rows += [row("measured", status="error", http_status=503)]
        rows += [row("measured", status="canceled")]
        rows += [row("measured") for _ in range(REPEATS - 3)]
        summary = _summary(rows, 1.0)
        self.assertEqual(summary["requests"], REPEATS)
        self.assertEqual(summary["successes"], REPEATS - 3)
        self.assertEqual(summary["http_429"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["canceled"], 1)
        self.assertEqual(summary["successful_completion_tokens"], (REPEATS - 3) * 10)


class FailurePartitionTests(unittest.TestCase):
    def test_same_case_name_rows_partition_by_requested_model(self):
        rows = [failure_row("small", status="error", http_status=503),
                failure_row("large")]
        affected, unaffected = _partition_by_model(rows, "small")
        self.assertEqual(affected, [rows[0]])
        self.assertEqual(unaffected, [rows[1]])


if __name__ == "__main__":
    unittest.main()
