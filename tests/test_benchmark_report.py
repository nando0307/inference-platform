import copy
import unittest

from benchmarks.common import distribution
from benchmarks.report import build_report


class BenchmarkReportTests(unittest.TestCase):
    def setUp(self):
        def sample(request_id, phase, status, count, latency):
            return dict(case="short", phase=phase, status=status, request_id=request_id,
                        prompt_tokens=64 if status == "success" else None,
                        completion_tokens=count, e2e_ms=latency, first_text_ms=latency / 2)

        self.client = {
            "environment": {}, "declared_server_config": {}, "warmups": 1, "repeats": 4,
            "workload": {"model_revision": "revision"},
            "cases": [{"name": "short", "elapsed_seconds": 2}],
            "samples": [
                sample("warmup", "warmup", "success", 100, 1000),
                sample("one", "measured", "success", 1, 10),
                sample("three", "measured", "success", 3, 30),
                sample("failed", "measured", "error", None, 500),
                sample("gone", "measured", "canceled", None, 700),
            ],
        }
        self.server = [
            {"type": "startup", "runtime": {"model_revision": "revision"}},
            dict(type="generation", request_id="three", status="success", prompt_tokens=64,
                 completion_tokens=3, ttft_ms=6, tpot_ms=4, inter_token_ms=[3, 5],
                 generation_ms=15, process_peak_rss_bytes=200),
            dict(type="generation", request_id="one", status="success", prompt_tokens=64,
                 completion_tokens=1, ttft_ms=2, tpot_ms=None, inter_token_ms=[],
                 generation_ms=3, process_peak_rss_bytes=100),
            dict(type="generation", request_id="gone", status="success", prompt_tokens=64,
                 completion_tokens=10, ttft_ms=20, tpot_ms=5, inter_token_ms=[5] * 9,
                 generation_ms=65, process_peak_rss_bytes=300),
        ]

    def test_excludes_warmup_and_undelivered_work_from_useful_throughput(self):
        case = build_report(self.client, self.server)["cases"][0]
        self.assertEqual((case["requests"], case["successes"], case["errors"], case["canceled"]),
                         (4, 2, 1, 1))
        self.assertEqual(case["successful_completion_tokens"], 4)
        self.assertEqual(case["useful_output_tokens_per_second"], 2)
        self.assertEqual(case["e2e_ms"]["p50"], 20)
        self.assertEqual(case["e2e_ms"]["p95"], 29)
        self.assertEqual(case["model_ttft_ms"]["p50"], 4)
        self.assertEqual(case["model_tpot_ms"]["n"], 1)
        self.assertEqual(case["model_tpot_ms"]["p50"], 4)
        self.assertEqual(case["inter_token_ms"]["n"], 2)
        self.assertEqual(case["inter_token_ms"]["p50"], 4)
        self.assertEqual(case["known_failed_or_canceled_generation_ms"], 65)
        self.assertEqual(case["process_peak_rss_bytes"], 300)

    def test_missing_model_observations_are_unavailable_not_zero(self):
        case = build_report(self.client, [])["cases"][0]
        self.assertEqual(case["missing_successful_model_samples"], 2)
        self.assertEqual(case["model_samples"], 0)
        self.assertIsNone(case["model_ttft_ms"]["p50"])
        self.assertIsNone(case["model_tpot_ms"]["p95"])
        self.assertIsNone(case["process_peak_rss_bytes"])

    def test_mismatched_revision_counts_and_duplicate_ids_cannot_form_a_report(self):
        for corruption in ("revision", "counts", "duplicate"):
            with self.subTest(corruption=corruption):
                records = copy.deepcopy(self.server)
                if corruption == "revision":
                    records[0]["runtime"]["model_revision"] = "different"
                elif corruption == "counts":
                    records[1]["completion_tokens"] = 4
                else:
                    records.append(records[1])
                with self.assertRaises(ValueError):
                    build_report(self.client, records)

    def test_interrupted_warmup_has_no_invented_measured_throughput(self):
        self.client["samples"] = [self.client["samples"][0]]
        self.client["samples"][0]["status"] = "canceled"
        self.client["cases"][0]["elapsed_seconds"] = 0
        self.client["interrupted"] = True
        report = build_report(self.client, [])
        case = report["cases"][0]
        self.assertTrue(report["interrupted"])
        self.assertEqual(case["requests"], 0)
        self.assertEqual(case["warmup_errors"], 1)
        self.assertIsNone(case["useful_output_tokens_per_second"])
        self.assertIsNone(case["e2e_ms"]["p50"])

    def test_quantiles_retain_real_zero_and_distinguish_missing_samples(self):
        self.assertEqual(distribution([None, 0, 10]),
                         dict(n=2, p50=5, p95=9.5, min=0, max=10))
        self.assertEqual(distribution([7])["p95"], 7)
        self.assertIsNone(distribution([None])["p95"])
        with self.assertRaises(ValueError):
            distribution([float("nan")])


if __name__ == "__main__":
    unittest.main()
