import copy
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.gpu import CASE, run_samples, select_case, summarize
from src.inference import GenerationResult, MODEL_ID
from test_benchmark_timing import tiny_tokenizer


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(
            tokenizer=tiny_tokenizer(), context_limit=2048,
            benchmark_model_revision="recorded-revision",
        )
        self.manifest = {
            "schema_version": 1, "model": MODEL_ID,
            "model_revision": "recorded-revision", "tokenizer_revision": "recorded-revision",
            "generation": {"temperature": 0, "top_p": 1},
            "baseline": {
                "device": "cpu", "dtype": "float32", "workers": 1,
                "concurrency": 1, "context_limit": 2048, "eos_counted": True,
            },
            "chat_template_sha256": hashlib.sha256(
                self.runtime.tokenizer.chat_template.encode()
            ).hexdigest(),
            "cases": [{
                "name": CASE, "messages": [{"role": "user", "content": "answer " * 57}],
                "prompt_tokens": 59, "max_tokens": 128,
            }],
        }

    def test_workload_mismatch_cannot_enter_measurement(self):
        case = select_case(self.manifest, self.runtime)
        self.assertEqual(case["prompt_tokens"], 59)
        for key, value in (
            ("model_revision", "different"), ("tokenizer_revision", "different"),
            ("chat_template_sha256", "different"), ("chat_template_sha256", None),
            ("generation", {"temperature": 1, "top_p": 1}),
        ):
            with self.subTest(key=key, value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest[key] = value
                with self.assertRaises(ValueError):
                    select_case(manifest, self.runtime)
        for key, value in (("prompt_tokens", 58), ("max_tokens", 127)):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.manifest)
                manifest["cases"][0][key] = value
                with self.assertRaises(ValueError):
                    select_case(manifest, self.runtime)


class WarmMeasurementTests(unittest.TestCase):
    case = {
        "name": CASE, "messages": [{"role": "user", "content": "a synthetic prompt"}],
        "prompt_tokens": 59, "max_tokens": 128,
    }
    result = GenerationResult("answer", 59, [10, 2], "stop")

    def test_clock_excludes_leading_fence_and_includes_pending_gpu_work(self):
        now = 0.0
        pending = 100.0  # Earlier initialization must never enter a measured interval.

        def synchronize():
            nonlocal now, pending
            now += pending + 0.25
            pending = 0.0

        def generate(*args, **kwargs):
            nonlocal now, pending
            now += 1.0  # Host work returns before the queued GPU work completes.
            pending = 2.0
            return self.result

        with (
            patch("torch.cuda.synchronize", side_effect=synchronize),
            patch("benchmarks.gpu.perf_counter", side_effect=lambda: now),
            patch("src.inference.generate", side_effect=generate),
        ):
            rows = run_samples(object(), self.case)
        self.assertEqual([row["phase"] for row in rows], ["warmup"] * 2 + ["measured"] * 5)
        self.assertEqual([row["warm_inference_ms"] for row in rows], [None, None] + [3250.0] * 5)
        summary = summarize(rows)
        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["completion_tokens_per_second"], 2 / 3.25)

    def test_failed_warmup_or_generation_stops_without_fabricated_usage(self):
        for successes_before_failure in (0, 3):
            with (
                self.subTest(successes_before_failure=successes_before_failure),
                patch("torch.cuda.synchronize"),
                patch("src.inference.generate", side_effect=(
                    [self.result] * successes_before_failure + [RuntimeError("private failure")]
                )),
            ):
                rows = run_samples(object(), self.case)
            self.assertEqual(len(rows), successes_before_failure + 1)
            self.assertEqual(rows[-1]["status"], "error")
            for key in ("warm_inference_ms", "prompt_tokens", "completion_tokens", "finish_reason"):
                self.assertIsNone(rows[-1][key])
            self.assertNotIn("private failure", str(rows))
            summary = summarize(rows)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["successful_measurements"], max(0, successes_before_failure - 2))
            self.assertEqual(summary["warmup_errors"] + summary["measured_errors"], 1)

    def test_throughput_weights_actual_tokens_by_successful_measured_time(self):
        rows = [
            {"phase": "warmup", "status": "success", "warm_inference_ms": None, "completion_tokens": 128},
            {"phase": "measured", "status": "success", "warm_inference_ms": 1000, "completion_tokens": 2},
            {"phase": "measured", "status": "success", "warm_inference_ms": 3000, "completion_tokens": 10},
            {"phase": "measured", "status": "error", "warm_inference_ms": None, "completion_tokens": None},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["completion_tokens_per_second"], 3)
        self.assertEqual(summary["median_warm_inference_ms"], 2000)
        self.assertEqual(summary["measured_errors"], 1)
        self.assertFalse(summary["complete"])
        empty = summarize([])
        self.assertIsNone(empty["median_warm_inference_ms"])
        self.assertIsNone(empty["completion_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
