import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.workload_generator.compare import compare, main
from benchmarks.workload_generator.trace import write_trace


class BenchmarkSuiteTests(unittest.TestCase):
    def setUp(self):
        self.trace = [dict(trace_schema_version=1, request_id=f"r{i}", trace_id="seed-7", scenario_id="s", phase_id="p",
                           model="m", model_revision="rev", scheduled_offset_s=i, stream=True)
                      for i in range(7)]
        self.meta = dict(trace_schema_version=1, trace_id="seed-7", request_count=7,
                         models={"m": {"id": "m", "revision": "rev"}})
        self.baseline = [dict(request_id=row["request_id"], model="m", model_revision="rev",
                              status=status, completion_ms=10 if status == "success" else None,
                              first_text_ms=2 if status == "success" else None,
                              scheduling_delay_ms=0 if status != "not_submitted" else None,
                              completion_tokens=4 if status == "success" else None,
                              prompt_tokens=3 if status == "success" else None,
                              error_code="overloaded" if status == "rejected" else None)
                         for row, status in zip(self.trace, ("success", "success", "rejected", "canceled",
                                                              "error", "timeout", "not_submitted"))]
        self.candidate = copy.deepcopy(self.baseline[:-1])
        self.candidate[1]["completion_ms"] = None
        self.candidate[2]["http_status"] = 429

    def test_counts_failures_missing_observations_and_deltas_without_inventing_zero(self):
        report = compare(self.trace, self.meta, self.baseline, self.candidate)
        group = report["groups"][0]
        self.assertEqual(group["baseline"]["scheduled"], 7)
        self.assertEqual(group["baseline"]["statuses"], dict(success=2, error=1, rejected=1,
                                                                 canceled=1, timeout=1, not_submitted=1))
        self.assertEqual(group["candidate"]["missing_results"], 1)
        self.assertEqual(group["candidate"]["metrics"]["completion_ms"]["n"], 1)
        self.assertEqual(group["candidate_minus_baseline"]["completion_ms"]["p50"], 0)
        self.assertEqual(group["candidate"]["metrics"]["scheduling_delay_ms"]["n"], 6)
        self.assertEqual(report["samples"][2]["candidate"]["http_status"], 429)
        self.assertEqual(report["samples"][2]["baseline"]["error_code"], "overloaded")
        self.assertIsNone(report["samples"][6]["candidate"])
        self.assertTrue(any("missing completion_ms" in warning for warning in report["warnings"]))
        self.assertTrue(any("missing result observations" in warning for warning in report["warnings"]))

    def test_revisions_and_identity_disagreements_are_rejected(self):
        for change in ("model_revision", "model", "trace_id", "phase_id", "scheduled_offset_s"):
            with self.subTest(change=change):
                rows = copy.deepcopy(self.baseline)
                rows[0][change] = "different"
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    compare(self.trace, self.meta, rows, self.candidate)
        trace = copy.deepcopy(self.trace)
        trace[0]["model_revision"] = "different"
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            compare(trace, self.meta, self.baseline, self.candidate)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare(self.trace, self.meta, self.baseline + [self.baseline[0]], self.candidate)
        with self.assertRaisesRegex(ValueError, "unknown"):
            compare(self.trace, self.meta, self.baseline + [{"request_id": "foreign"}], self.candidate)

    def test_cli_is_byte_stable_and_does_not_overwrite_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            write_trace(self.trace, self.meta, trace)
            baseline, candidate = (Path(directory) / name for name in ("baseline.jsonl", "candidate.jsonl"))
            for path, rows in ((baseline, self.baseline), (candidate, self.candidate)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            first, second = (Path(directory) / name for name in ("one.json", "two.json"))
            for output in (first, second):
                self.assertEqual(main(["--trace", str(trace), "--baseline", str(baseline),
                                       "--candidate", str(candidate), "--output", str(output)]), 0)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with self.assertRaises(FileExistsError):
                main(["--trace", str(trace), "--baseline", str(baseline),
                      "--candidate", str(candidate), "--output", str(first)])


if __name__ == "__main__":
    unittest.main()
