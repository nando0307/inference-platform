import unittest

from benchmarks.client import _summary
from benchmarks.multi_gpu_serving import REPEATS, WARMUPS, _lifecycle, _partition_by_model


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
        self.assertEqual(summary["measured_requests"], REPEATS)
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
        self.assertEqual(summary["measured_requests"], REPEATS)
        self.assertEqual(summary["successes"], REPEATS - 3)
        self.assertEqual(summary["rejections"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["canceled"], 1)
        self.assertEqual(summary["successful_completion_tokens"], (REPEATS - 3) * 10)


class LifecycleEvidenceTests(unittest.TestCase):
    metrics = {"uptime_seconds": 42.5,
               "workers": {"small-0": {"ready_seconds_after_start": 12.5},
                           "large-1": {"ready_seconds_after_start": 20.0}}}
    workers = ["small-0", "large-1"]

    def shutdown(self, exit_status=0, small=True, large=True):
        return {"gateway_exit_status": exit_status,
                "workers_stopped": {"small-0": small, "large-1": large}}

    def test_orderly_shutdown_with_every_worker_stopped_completes_the_artifact(self):
        lifecycle = _lifecycle(self.metrics, self.workers, self.shutdown())
        self.assertTrue(lifecycle["complete"])
        self.assertEqual(lifecycle["startup"]["gateway_uptime_seconds"], 42.5)
        self.assertEqual(lifecycle["startup"]["worker_ready_seconds_after_start"],
                         {"small-0": 12.5, "large-1": 20.0})
        self.assertEqual(lifecycle["cleanup"]["gateway_exit_status"], 0)
        self.assertTrue(all(lifecycle["cleanup"]["workers_stopped"].values()))

    def test_unconfirmed_worker_teardown_or_crashed_gateway_blocks_completion(self):
        self.assertFalse(_lifecycle(self.metrics, self.workers,
                                    self.shutdown(large=False))["complete"])
        self.assertFalse(_lifecycle(self.metrics, self.workers,
                                    self.shutdown(exit_status=1))["complete"])
        self.assertFalse(_lifecycle(self.metrics, self.workers,
                                    self.shutdown(exit_status=None))["complete"])

    def test_empty_or_unknown_worker_set_never_completes(self):
        self.assertFalse(_lifecycle(self.metrics, [],
                                    {"gateway_exit_status": 0, "workers_stopped": {}})["complete"])
        self.assertFalse(_lifecycle(self.metrics, self.workers,
                                    {"gateway_exit_status": 0, "workers_stopped": {}})["complete"])
        self.assertFalse(_lifecycle({"workers": {}}, self.workers, self.shutdown())["complete"])
        self.assertFalse(_lifecycle(
            {"uptime_seconds": 1.0, "workers": {"small-0": {}}},
            self.workers, self.shutdown())["complete"])


class FailurePartitionTests(unittest.TestCase):
    def test_same_case_name_rows_partition_by_requested_model(self):
        rows = [failure_row("small", status="error", http_status=503),
                failure_row("large")]
        affected, unaffected = _partition_by_model(rows, "small")
        self.assertEqual(affected, [rows[0]])
        self.assertEqual(unaffected, [rows[1]])


if __name__ == "__main__":
    unittest.main()
