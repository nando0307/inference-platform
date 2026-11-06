import io
import json
import os
import unittest
import weakref
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import GenerationConfig, Qwen2Config, Qwen2ForCausalLM, StaticCache

from benchmarks.cache_scaling import main, measure, new_result
from src.batched_generation import generate_batch
from src.cached_generation import generate_cached
from src.inference import Runtime
from test_benchmark_timing import tiny_tokenizer


class CachedGenerationTests(unittest.TestCase):
    def setUp(self):
        tokenizer = tiny_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=1024, eos_token_id=[2, 6], pad_token_id=2,
            )).eval()
        self.runtime = Runtime(model, tokenizer, {2, 6}, context_limit=1024)
        self.runtime.benchmark_model_revision = "tiny-test-model"

    def inputs(self, conversations):
        tokenizer = self.runtime.tokenizer
        texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 for messages in conversations]
        return tokenizer(texts, padding=True, padding_side="left", return_tensors="pt",
                         add_special_tokens=False, return_attention_mask=True)

    def test_independent_eos_dense_cache_and_last_selected_token(self):
        conversations = [[{"role": "user", "content": prompt}] for prompt in ("answer", "tail tail tail")]
        inputs = self.inputs(conversations)
        step = 0
        short_rows = None

        def choose_tokens(model, args, kwargs, output):
            nonlocal step, short_rows
            if kwargs["input_ids"].shape[1] > 1:
                step = 0
                short_rows = (kwargs["input_ids"] == 4).any(dim=1).tolist()
            output.logits.fill_(-20)
            for row, short in enumerate(short_rows):
                output.logits[row, :, 2 if short else 6 if step == 3 else 5] = 20
            step += 1

        hook = self.runtime.model.register_forward_hook(choose_tokens, with_kwargs=True)
        self.addCleanup(hook.remove)
        for cap, second_ids, finish in ((1, [5], "length"), (3, [5, 5, 5], "length"), (4, [5, 5, 5, 6], "stop")):
            with self.subTest(cap=cap):
                result = generate_cached(self.runtime.model, inputs, max_new_tokens=cap, pad_token_id=2)
                reference = generate_batch(self.runtime, conversations, max_new_tokens=cap)
                self.assertEqual(result["generated_ids"], [[2], second_ids])
                self.assertEqual(result["generated_ids"], [row.generated_ids for row in reference])
                self.assertEqual(result["finish_reasons"], ["stop", finish])
                self.assertEqual(result["prompt_tokens"], [3, 5])
                for snapshot, positions in ((result["prefill"], 5), (result["final"], 5 + cap - 1)):
                    self.assertEqual(snapshot["cached_positions"], positions)
                    self.assertEqual(snapshot["first_key_shape"], [2, 2, positions, 4])
                    self.assertEqual(snapshot["measured_bytes"], 256 * positions)
                    self.assertEqual(snapshot["predicted_bytes"], 256 * positions)
        # All rows finish on their first selection: no decode forward or cache growth.
        all_short = self.inputs([conversations[0], conversations[0]])
        result = generate_cached(self.runtime.model, all_short, max_new_tokens=4, pad_token_id=2)
        self.assertEqual(result["generated_ids"], [[2], [2]])
        self.assertEqual(result["final"]["cached_positions"], 3)
        self.assertEqual(result["decode_ms"], 0)

    def test_fixed_length_suppresses_all_eos_without_changing_natural_stopping(self):
        inputs = self.inputs([[{"role": "user", "content": text}] for text in ("answer", "tail tail tail")])
        model = self.runtime.model

        def prefer_eos(model, args, output):
            output.logits.fill_(-20)
            output.logits[:, :, 2] = 30
            output.logits[:, :, 6] = 20
            output.logits[:, :, 5] = 10

        hook = model.register_forward_hook(prefer_eos)
        self.addCleanup(hook.remove)
        for cap, capacity in ((1, None), (4, None), (4, 12)):
            with self.subTest(cap=cap, capacity=capacity):
                fixed = generate_cached(
                    model, inputs, max_new_tokens=cap, pad_token_id=2,
                    cache_capacity=capacity, fixed_length=True,
                )
                with torch.inference_mode():
                    reference = model.generate(**inputs, generation_config=GenerationConfig(
                        do_sample=False, min_new_tokens=cap, max_new_tokens=cap,
                        eos_token_id=[2, 6], pad_token_id=2,
                    ))
                self.assertEqual(fixed["generated_ids"], [[5] * cap, [5] * cap])
                self.assertEqual(fixed["generated_ids"], reference[:, inputs["input_ids"].shape[1]:].tolist())
                self.assertEqual(fixed["finish_reasons"], ["length", "length"])
                self.assertEqual(fixed["final"]["cached_positions"], 5 + cap - 1)
                natural = generate_cached(model, inputs, max_new_tokens=cap, pad_token_id=2)
                self.assertEqual(natural["generated_ids"], [[2], [2]])
                self.assertEqual(natural["finish_reasons"], ["stop", "stop"])
                self.assertEqual(natural["final"]["cached_positions"], 5)

    def test_real_greedy_parity_with_left_padding_and_batch_scaling(self):
        self.runtime.model.generation_config.eos_token_id = None
        self.runtime.eos_token_ids = set()
        for size in (1, 2, 4):
            conversations = [[{"role": "user", "content": "tail " * (index + 1)}] for index in range(size)]
            inputs = self.inputs(conversations)
            before = inputs["attention_mask"].clone()
            result = generate_cached(self.runtime.model, inputs, max_new_tokens=4, pad_token_id=2)
            reference = generate_batch(self.runtime, conversations, max_new_tokens=4)
            self.assertEqual(result["generated_ids"], [row.generated_ids for row in reference])
            self.assertEqual(result["finish_reasons"], ["length"] * size)
            self.assertTrue(torch.equal(inputs["attention_mask"], before))
            width = inputs["input_ids"].shape[1]
            self.assertEqual(result["prefill"]["measured_bytes"], 128 * size * width)
            self.assertEqual(result["final"]["measured_bytes"], 128 * size * (width + 3))

    def test_static_capacity_masks_unused_values_and_preserves_logits(self):
        self.runtime.model.generation_config.eos_token_id = None
        self.runtime.eos_token_ids = set()
        conversations = [[{"role": "user", "content": text}] for text in ("answer", "tail tail tail")]
        inputs = self.inputs(conversations)
        captured = []

        def inspect_and_poison(model, args, kwargs, output):
            captured.append(output.logits[:, -1, :].clone())
            cache = output.past_key_values
            if isinstance(cache, StaticCache):
                used = kwargs["attention_mask"].shape[1]
                # Unused values must remain invisible even when they are not zero.
                # Padding also has zero keys in this tiny model, so writes must
                # not infer their offset by counting nonzero keys.
                for layer in cache.layers:
                    layer.values[:, :, used:, :].fill_(10000)

        hook = self.runtime.model.register_forward_hook(inspect_and_poison, with_kwargs=True)
        self.addCleanup(hook.remove)
        baseline = generate_cached(self.runtime.model, inputs, max_new_tokens=4, pad_token_id=2)
        baseline_logits = captured[:]
        for capacity in (8, 16):
            captured.clear()
            result = generate_cached(
                self.runtime.model, inputs, max_new_tokens=4, pad_token_id=2, cache_capacity=capacity,
            )
            self.assertEqual(result["generated_ids"], baseline["generated_ids"])
            self.assertEqual(result["finish_reasons"], ["length", "length"])
            for actual, expected in zip(captured, baseline_logits, strict=True):
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            for snapshot, used in ((result["prefill"], 5), (result["final"], 8)):
                self.assertEqual(snapshot["cached_positions"], used)
                self.assertEqual(snapshot["allocated_positions"], capacity)
                self.assertEqual(snapshot["first_key_shape"], [2, 2, capacity, 4])
                self.assertEqual(snapshot["predicted_used_bytes"], 256 * used)
                self.assertEqual(snapshot["predicted_bytes"], 256 * capacity)
                self.assertEqual(snapshot["measured_bytes"], 256 * capacity)

    def test_static_eos_does_not_fill_capacity_and_invalid_capacity_is_rejected(self):
        inputs = self.inputs([[{"role": "user", "content": "answer"}]])

        def eos(model, args, kwargs, output):
            output.logits.fill_(-20)
            output.logits[:, :, 2] = 20

        hook = self.runtime.model.register_forward_hook(eos, with_kwargs=True)
        try:
            result = generate_cached(
                self.runtime.model, inputs, max_new_tokens=4, pad_token_id=2, cache_capacity=6,
            )
        finally:
            hook.remove()
        self.assertEqual(result["generated_ids"], [[2]])
        self.assertEqual(result["finish_reasons"], ["stop"])
        self.assertEqual(result["final"]["cached_positions"], 3)
        self.assertEqual(result["final"]["measured_bytes"], 128 * 6)
        self.assertEqual(result["final"]["predicted_used_bytes"], 128 * 3)
        self.assertEqual(result["decode_ms"], 0)
        with patch.object(self.runtime.model, "forward", side_effect=AssertionError("Must not run")):
            for capacity in (True, 0, 5, 6.5, 1025):
                with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                    generate_cached(
                        self.runtime.model, inputs, max_new_tokens=4, pad_token_id=2, cache_capacity=capacity,
                    )

    def test_cpu_allocation_comparison_keeps_used_bytes_equal_but_capacity_fixed(self):
        self.runtime.model.generation_config.eos_token_id = None
        self.runtime.eos_token_ids = set()
        record = measure(self.runtime, new_result(max_new_tokens=4, compare_static=True))
        self.assertEqual(record["status"], "completed")
        dynamic, small, large = [case["manual"] for case in record["cases"]]
        for case in record["cases"]:
            self.assertTrue(all(case["checks"].values()))
        for static, capacity in ((small, 160), (large, 256)):
            for stage in ("prefill", "final"):
                self.assertEqual(static[stage]["measured_bytes"], 128 * capacity)
                self.assertEqual(static[stage]["predicted_used_bytes"], dynamic[stage]["measured_bytes"])
        self.assertEqual(
            dynamic["final"]["measured_bytes"] - dynamic["prefill"]["measured_bytes"], 3 * 128,
        )
        with self.assertRaises(ValueError):
            new_result(max_new_tokens=17, compare_static=True)
        runtime = SimpleNamespace(model=SimpleNamespace(device=torch.device("cuda"), dtype=torch.float32))
        with self.assertRaises(ValueError):
            measure(runtime, new_result(compare_static=True))
        with TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["--compare-static", "--device", "cuda", "--output", str(Path(directory) / "no-gpu.json")])
            self.assertEqual(raised.exception.code, 2)

    def test_invalid_masks_caps_and_context_are_rejected_before_forward(self):
        inputs = self.inputs([[{"role": "user", "content": "answer"}]])
        with patch.object(self.runtime.model, "forward", side_effect=AssertionError("Must not run")):
            for cap in (0, True, 513):
                with self.assertRaises(ValueError):
                    generate_cached(self.runtime.model, inputs, max_new_tokens=cap, pad_token_id=2)
            for mask in (torch.tensor([[1, 1, 0]]), torch.zeros((1, 3)), torch.tensor([[1, 2, 1]])):
                with self.assertRaises(ValueError):
                    generate_cached(self.runtime.model, {**inputs, "attention_mask": mask},
                                    max_new_tokens=1, pad_token_id=2)
            self.runtime.model.config.max_position_embeddings = 3
            with self.assertRaises(ValueError):
                generate_cached(self.runtime.model, inputs, max_new_tokens=1, pad_token_id=2)

    def test_five_case_measurement_and_mismatches_affect_status(self):
        self.runtime.model.generation_config.eos_token_id = None
        self.runtime.eos_token_ids = set()
        with (
            patch("torch.cuda.synchronize", side_effect=AssertionError("CPU touched CUDA")),
            patch("torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("CPU touched CUDA")),
            patch("torch.cuda.max_memory_allocated", side_effect=AssertionError("CPU touched CUDA")),
            patch("torch.cuda.max_memory_reserved", side_effect=AssertionError("CPU touched CUDA")),
        ):
            record = measure(self.runtime, new_result(max_new_tokens=1))
        self.assertEqual(record["status"], "completed")
        medium, twice, four_times = [record["cases"][index]["manual"]["prefill"]["measured_bytes"] for index in (1, 3, 4)]
        self.assertEqual(twice, medium * 2)
        self.assertEqual(four_times, medium * 4)
        for case in record["cases"]:
            self.assertTrue(all(case["checks"].values()))
            self.assertEqual(case["manual"]["prefill"], case["manual"]["final"])
            self.assertIsNone(case["peak_allocated_bytes"])
            self.assertIsNone(case["peak_reserved_bytes"])
        original = generate_batch

        def different_reference(*args, **kwargs):
            rows = original(*args, **kwargs)
            rows[0].generated_ids = [-1]
            return rows

        with patch("src.batched_generation.generate_batch", side_effect=different_reference):
            mismatch = measure(self.runtime, new_result(max_new_tokens=1))
        self.assertEqual(mismatch["status"], "failed")
        self.assertTrue(all(case["status"] == "mismatch" for case in mismatch["cases"]))

    def test_cuda_peaks_exclude_warmup_and_reference_but_retain_reservations(self):
        # Real CPU Qwen forwards with a simulated asynchronous CUDA allocator.
        # Wrong fences/reset/release boundaries change the reported byte counts.
        model = self.runtime.model
        model.generation_config.eos_token_id = None
        self.runtime.eos_token_ids = set()
        for cap in (1, 3):
            state = {
                "pending": [], "peak": 100, "reserved": 100, "reserved_peak": 100,
                "phase": "warmup", "warmup": None, "forwards": 0, "references": 0,
            }

            def synchronize(device):
                for allocation in state["pending"]:
                    state["peak"] = max(state["peak"], allocation)
                    state["reserved"] = max(state["reserved"], allocation)
                    state["reserved_peak"] = max(state["reserved_peak"], state["reserved"])
                state["pending"].clear()

            def reset(device):
                # Warm-up tensors must be released; model/input allocations remain.
                state["peak"] = 100 if state["warmup"]() is None else 900
                state["reserved_peak"] = state["reserved"]
                state["phase"] = "manual"
                state["forwards"] = 0

            class SimulatedCudaModel:
                device = torch.device("cuda:0")
                dtype = model.dtype
                config = model.config
                generation_config = model.generation_config

                def __call__(self, **kwargs):
                    output = model(**kwargs)
                    if state["phase"] == "warmup":
                        state["warmup"] = weakref.ref(output)
                        state["pending"].append(900)
                    else:
                        state["forwards"] += 1
                        state["pending"].append(100 + 100 * state["forwards"])
                    return output

                def generate(self, **kwargs):
                    output = model.generate(**kwargs)
                    state["pending"].append(2000 + 1000 * state["references"])
                    synchronize(self.device)
                    state["references"] += 1
                    state["phase"] = "warmup"
                    return output

            self.runtime.model = SimulatedCudaModel()
            with (
                self.subTest(cap=cap),
                patch("transformers.tokenization_utils_base.BatchEncoding.to", lambda inputs, *args, **kwargs: inputs),
                patch("torch.cuda.get_device_name", return_value="simulated CUDA"),
                patch("torch.cuda.synchronize", side_effect=synchronize),
                patch("torch.cuda.reset_peak_memory_stats", side_effect=reset),
                patch("torch.cuda.max_memory_allocated", side_effect=lambda device: state["peak"]),
                patch("torch.cuda.max_memory_reserved", side_effect=lambda device: state["reserved_peak"]),
                patch("torch.cuda.empty_cache", side_effect=AssertionError("Reservations must persist")),
            ):
                record = measure(self.runtime, new_result(max_new_tokens=cap))
            self.assertEqual(record["status"], "completed")
            for case, reserved in zip(record["cases"], (900, 2000, 3000, 4000, 5000)):
                self.assertEqual(case["peak_allocated_bytes"], 100 + 100 * cap)
                self.assertEqual(case["peak_reserved_bytes"], reserved)
                self.assertTrue(all(case["checks"].values()))

    def test_unsupported_device_precision_and_unavailable_cuda_do_not_fall_back(self):
        for device, dtype in (("mps", torch.float32), ("cpu", torch.float16)):
            runtime = SimpleNamespace(model=SimpleNamespace(device=torch.device(device), dtype=dtype))
            with self.subTest(device=device, dtype=dtype), self.assertRaises(ValueError):
                measure(runtime, new_result())
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, HF_HUB_OFFLINE="1"),
            patch("torch.cuda.is_available", return_value=False),
            patch("src.inference.load_runtime", side_effect=AssertionError("Must not load a CPU fallback")),
            redirect_stdout(io.StringIO()),
        ):
            output = Path(directory) / "unavailable.json"
            self.assertEqual(main(["--device", "cuda", "--output", str(output)]), 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exception_type"], "RuntimeError")
            self.assertIsNone(result["runtime"])
            self.assertTrue(all(case["status"] == "not_run" for case in result["cases"]))

    def test_failure_preserves_measurement_and_refuses_overwrite(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false"),
            patch("src.inference.load_runtime", return_value=self.runtime),
            patch("src.batched_generation.generate_batch", side_effect=RuntimeError("private detail")),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            output = Path(directory) / "result.json"
            args = ["--output", str(output), "--max-new-tokens", "1"]
            self.assertEqual(main(args), 1)
            saved = output.read_bytes()
            result = json.loads(saved)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["cases"][0]["status"], "error")
            self.assertEqual(result["cases"][0]["exception_type"], "RuntimeError")
            manual = result["cases"][0]["manual"]
            self.assertEqual(manual["prefill"]["measured_bytes"], 128 * sum(manual["prompt_tokens"]))
            self.assertIsNone(result["cases"][0]["reference"])
            self.assertTrue(all(case["status"] == "not_run" for case in result["cases"][1:]))
            self.assertNotIn(b"private detail", saved)
            with self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
