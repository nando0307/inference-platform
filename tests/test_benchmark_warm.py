import copy
import hashlib
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from benchmarks.warm import CASE, main, run_samples, select_case, summarize
from src.inference import GenerationResult, MODEL_ID, Runtime
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

    def test_cpu_cli_saves_failure_and_refuses_overwrite(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, HF_HUB_OFFLINE="1"),
            patch("src.inference.load_runtime", side_effect=RuntimeError("private failure detail")),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            workload = Path(directory) / "workload.json"
            output = Path(directory) / "result.json"
            workload.write_text(json.dumps(self.manifest), encoding="utf-8")
            args = ["--workload", str(workload), "--output", str(output)]
            self.assertEqual(main(args), 1)
            saved = output.read_bytes()
            result = json.loads(saved)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exception_type"], "RuntimeError")
            self.assertFalse(result["samples_available"])
            self.assertIsNone(result["summary"])
            self.assertNotIn(b"private failure detail", saved)
            with self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), saved)


class WarmMeasurementTests(unittest.TestCase):
    case = {
        "name": CASE, "messages": [{"role": "user", "content": "a synthetic prompt"}],
        "prompt_tokens": 59, "max_tokens": 128,
    }
    result = GenerationResult("answer", 59, [10, 2], "stop")
    cuda_runtime = SimpleNamespace(model=SimpleNamespace(device=torch.device("cuda:0")))

    def test_clock_excludes_leading_fence_and_includes_pending_gpu_work(self):
        now = 0.0
        pending = 100.0  # Earlier initialization must never enter a measured interval.

        def synchronize(device):
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
            patch("benchmarks.warm.perf_counter", side_effect=lambda: now),
            patch("src.inference.generate", side_effect=generate),
        ):
            rows = run_samples(self.cuda_runtime, self.case)
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
                rows = run_samples(self.cuda_runtime, self.case)
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

    def test_cpu_runs_real_generation_without_touching_cuda(self):
        tokenizer = tiny_tokenizer()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=128, n_embd=8, n_layer=1,
                n_head=1, bos_token_id=None, eos_token_id=2, pad_token_id=0,
            )).eval()
        runtime = Runtime(model, tokenizer, {2}, context_limit=2048)
        case = {**self.case, "messages": [{"role": "user", "content": "answer " * 57}]}

        def prefer_eos(model, args, output):
            output.logits.fill_(-10)
            output.logits[..., 2] = 10

        hook = model.register_forward_hook(prefer_eos)
        try:
            with patch("torch.cuda.synchronize", side_effect=AssertionError("CPU touched CUDA")):
                rows = run_samples(runtime, case)
        finally:
            hook.remove()
        self.assertTrue(summarize(rows)["complete"])
        self.assertTrue(all(row["prompt_tokens"] == 59 for row in rows))
        self.assertTrue(all(row["completion_tokens"] == 1 and row["finish_reason"] == "stop" for row in rows))

    def test_other_accelerators_cannot_silently_use_cpu_timing(self):
        runtime = SimpleNamespace(model=SimpleNamespace(device=torch.device("mps")))
        with self.assertRaises(ValueError):
            run_samples(runtime, self.case)


if __name__ == "__main__":
    unittest.main()
