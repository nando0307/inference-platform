import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from benchmarks.batching import PROMPTS, measure, new_result
from src.inference import GenerationResult, Runtime
from test_benchmark_timing import tiny_tokenizer


def result_for(messages):
    index = PROMPTS.index(messages[0]["content"])
    return GenerationResult(f"output {index}", 3 + index, [10 + index] * index + [2], "stop")


class BatchBenchmarkTests(unittest.TestCase):
    def test_real_cpu_workload_groups_and_actual_token_accounting(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            tokenizer = tiny_tokenizer()
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=32, n_embd=8, n_layer=1,
                n_head=1, bos_token_id=None, eos_token_id=2, pad_token_id=0,
            )).eval()
        runtime = Runtime(model, tokenizer, {2}, context_limit=32)
        runtime.benchmark_model_revision = "tiny-test-model"
        record = new_result(max_new_tokens=3, warmups=1, repeats=2)
        record["conversations"] = [[{"role": "user", "content": text}] for text in
                                   ("answer", "tail tail", "answer tail answer", "tail")]
        with (
            patch.dict(os.environ, TOKENIZERS_PARALLELISM="false"),
            patch("torch.cuda.synchronize", side_effect=AssertionError("CPU touched CUDA")),
        ):
            measure(runtime, record)
        self.assertEqual(record["status"], "completed")
        for configuration, expected_groups in zip(record["configurations"],
                                                  ([[0], [1], [2], [3]], [[0, 1], [2, 3]], [[0, 1, 2, 3]])):
            with self.subTest(batch_size=configuration["batch_size"]):
                samples = configuration["samples"]
                self.assertEqual([sample["phase"] for sample in samples], ["warmup", "measured", "measured"])
                self.assertIsNone(samples[0]["workload_ms"])
                for sample in samples:
                    self.assertEqual([call["input_indices"] for call in sample["calls"]], expected_groups)
                    rows = [row for call in sample["calls"] for row in call["results"]]
                    self.assertEqual([row["prompt_tokens"] for row in rows], [3, 4, 5, 3])
                    self.assertEqual(sample["completion_tokens"], sum(len(row["generated_ids"]) for row in rows))
                    self.assertIsNone(sample["peak_allocated_bytes"])
                    self.assertIsNone(sample["peak_reserved_bytes"])
                summary = configuration["summary"]
                self.assertTrue(summary["complete"])
                self.assertAlmostEqual(summary["output_tokens_per_second"],
                                       sum(sample["completion_tokens"] for sample in samples[1:]) * 1000
                                       / sum(sample["workload_ms"] for sample in samples[1:]))

    def test_cuda_clock_boundaries_peak_reset_and_output_differences(self):
        now, pending = 0.0, 100.0
        peak = 900  # Earlier/warm-up allocations must not become measured allocated peaks.

        def synchronize(device):
            nonlocal now, pending
            now += pending + 0.25
            pending = 0.0

        def reset(device):
            nonlocal peak
            peak = 100  # Persistent model allocation remains in the absolute peak.

        def work(conversations):
            nonlocal now, pending, peak
            now += 1.0
            pending = 2.0
            peak = max(peak, 100 + 100 * len(conversations))
            results = [result_for(messages) for messages in conversations]
            if len(conversations) == 2:
                for result in results:
                    if result.prompt_tokens == 4:
                        result.generated_ids[0] = 99  # Same length, different output: retain and report it.
            return results

        runtime = SimpleNamespace(model=SimpleNamespace(device=torch.device("cuda:0")))
        record = new_result(max_new_tokens=4, warmups=1, repeats=2)
        with (
            patch("benchmarks.batching.describe_runtime", return_value={"batch_size": 1}),
            patch("benchmarks.batching.perf_counter", side_effect=lambda: now),
            patch("torch.cuda.synchronize", side_effect=synchronize),
            patch("torch.cuda.reset_peak_memory_stats", side_effect=reset),
            patch("torch.cuda.max_memory_allocated", side_effect=lambda device: peak),
            patch("torch.cuda.max_memory_reserved", return_value=1000),
            patch("src.inference.generate", side_effect=lambda runtime, messages, **kwargs: work([messages])[0]),
            patch("src.batched_generation.generate_batch", side_effect=lambda runtime, conversations, **kwargs: work(conversations)),
        ):
            measure(runtime, record)
        self.assertEqual(record["status"], "completed")
        for configuration, duration, memory, differences in zip(record["configurations"],
                                                                (14000, 7000, 3500), (200, 300, 500), (0, 2, 0)):
            summary = configuration["summary"]
            self.assertEqual(summary["median_call_ms"], 3250)
            self.assertEqual(summary["median_workload_ms"], duration)
            self.assertEqual(summary["completion_tokens"], 20)
            self.assertAlmostEqual(summary["output_tokens_per_second"], 10000 / duration)
            self.assertEqual(summary["peak_allocated_bytes"], memory)
            self.assertEqual(summary["peak_reserved_bytes"], 1000)
            self.assertEqual(summary["output_mismatches"], differences)
        changed = record["configurations"][1]["samples"][1]["calls"][0]["results"][1]
        self.assertFalse(changed["matches_reference"])
        self.assertEqual(changed["generated_ids"], [99, 2])

    def test_failed_or_interrupted_workload_preserves_calls_but_is_not_throughput(self):
        runtime = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
        for failure, status in ((RuntimeError("private detail"), "failed"), (KeyboardInterrupt(), "interrupted")):
            calls = 0

            def generate(runtime, messages, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 7:  # Four warm-up calls, two completed measured calls, then failure.
                    raise failure
                return result_for(messages)

            record = new_result(max_new_tokens=4, warmups=1, repeats=2)
            with (
                self.subTest(status=status),
                patch("benchmarks.batching.describe_runtime", return_value={"batch_size": 1}),
                patch("src.inference.generate", side_effect=generate),
            ):
                measure(runtime, record)
                self.assertEqual(record["status"], status)
                configuration = record["configurations"][0]
                sample = configuration["samples"][1]
                self.assertEqual([call["status"] for call in sample["calls"]],
                                 ["success", "success", "error" if status == "failed" else "interrupted", "not_run"])
                self.assertEqual([call["results"][0]["generated_ids"] for call in sample["calls"][:2]], [[2], [11, 2]])
                self.assertIsNone(sample["workload_ms"])
                self.assertIsNone(sample["completion_tokens"])
                self.assertEqual(configuration["summary"]["failed_measured_workloads"], 1)
                self.assertIsNone(configuration["summary"]["output_tokens_per_second"])
                self.assertIsNone(configuration["summary"]["median_call_ms"])
                self.assertFalse(configuration["summary"]["complete"])
                self.assertEqual(record["configurations"][1]["samples"], [])
                self.assertNotIn("private detail", json.dumps(record))

    def test_workload_bounds_and_unsupported_accelerator(self):
        for arguments in ({"max_new_tokens": 0}, {"max_new_tokens": 129}, {"max_new_tokens": True},
                          {"warmups": 0}, {"warmups": 3}, {"repeats": 0}, {"repeats": 6}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                new_result(**arguments)
        runtime = SimpleNamespace(model=SimpleNamespace(device=torch.device("mps")))
        with self.assertRaises(ValueError):
            measure(runtime, new_result())


if __name__ == "__main__":
    unittest.main()
