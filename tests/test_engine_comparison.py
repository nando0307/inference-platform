"""Offline contract checks, including the real API over a loopback socket."""

import asyncio
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from unittest.mock import patch

import httpx
import torch
from transformers import GenerationConfig, GPT2Config, GPT2LMHeadModel

from benchmarks.engine_comparison import (
    exercise_baseline, fixed_generation, main, measure_backend, new_backend_result,
)
from benchmarks.continuous_batching import _scenario
from src import api
from src.inference import Runtime, generate
from test_benchmark_timing import tiny_tokenizer
from test_continuous_batching import CASES, WORKLOAD, Stream, Tokenizer, event, terminal


def runtime():
    tokenizer = tiny_tokenizer()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=len(tokenizer), n_positions=256, n_embd=8, n_layer=1,
            n_head=1, bos_token_id=None, eos_token_id=2, pad_token_id=0,
        ))
    model.eval()
    return Runtime(model, tokenizer, {2}, context_limit=256)


def prefer_eos(model, args, output):
    # Actual generate() still applies EOS suppression, stopping and usage accounting.
    output.logits.fill_(-10)
    output.logits[..., 4] = 5
    output.logits[..., 2] = 10


class ComparisonTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_baseline_uses_only_accepted_fields_but_still_requires_fixed_counts(self):
        for completion_tokens in (16, 15):
            with self.subTest(completion_tokens=completion_tokens):
                async def body():
                    yield event({"choices": [{
                        "index": 0, "delta": {"content": "answer"}, "finish_reason": None,
                    }]})
                    yield terminal(completion_tokens, "length" if completion_tokens == 16 else "stop")

                def handler(request):
                    payload = json.loads(request.content)
                    # This is the production extra='forbid' trust boundary, not a field-list copy.
                    api.ChatCompletionRequest.model_validate(payload)
                    return httpx.Response(
                        200, headers={"content-type": "text/event-stream"}, stream=Stream(body()),
                    )

                scenario = {"name": "sequential"}
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    operation = _scenario(
                        client, "http://controlled.invalid", {**WORKLOAD, "model": api.MODEL_ID},
                        CASES[:1], scenario, vllm_options=False,
                    )
                    if completion_tokens == 16:
                        await operation
                        self.assertEqual(scenario["rows"][0]["completion_tokens"], 16)
                        self.assertIsNone(scenario["rows"][0]["checks"]["prompt_token_ids_match"])
                    else:
                        with self.assertRaisesRegex(RuntimeError, "request checks failed"):
                            await operation
                        self.assertFalse(scenario["rows"][0]["checks"]["fixed_completion_count_matches_cap"])

    async def test_absent_refill_overlap_does_not_fail_a_valid_baseline(self):
        calls = []

        async def body(count, reason):
            yield event({"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}]})
            yield terminal(count, reason)

        def handler(request):
            payload = json.loads(request.content)
            api.ChatCompletionRequest.model_validate(payload)
            calls.append(payload)
            natural = payload["messages"][0]["content"] == "Reply with exactly one word: Hello"
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"},
                stream=Stream(body(1 if natural else payload["max_tokens"], "stop" if natural else "length")),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        record = new_backend_result("transformers")
        with patch("httpx.AsyncClient", return_value=client):
            await measure_backend(
                "http://controlled.invalid", record, Tokenizer(), [99], fixed_scope=nullcontext,
            )
        self.assertEqual(len(calls), 14)
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["checks"]["all_passed"])
        self.assertTrue(record["checks"]["natural_eos_demonstrated"])
        self.assertFalse(record["diagnostics"]["refill_demonstrated"])
        self.assertNotIn("refill_demonstrated", record["checks"])
        self.assertEqual(
            [scenario["discarded"] for scenario in record["scenarios"]],
            [True, True, False, False, True, False],
        )


class FixedGenerationTests(unittest.TestCase):
    def test_fixed_scope_restores_config_after_success_and_model_failure(self):
        current = runtime()
        model = current.model
        defaults = model.generation_config.to_dict()
        original_generate = model.generate
        config = GenerationConfig(
            max_new_tokens=3, eos_token_id=2, pad_token_id=0, do_sample=False,
        )
        hook = model.register_forward_hook(prefer_eos)
        try:
            with fixed_generation(current):
                output = model.generate(input_ids=torch.tensor([[3, 6]]), generation_config=config)
            self.assertEqual(output[0, 2:].tolist(), [4, 4, 4])
            self.assertIsNone(config.min_new_tokens)

            def fail(model, args):
                raise RuntimeError("controlled forward failure")

            failure_hook = model.register_forward_pre_hook(fail)
            try:
                with self.assertRaisesRegex(RuntimeError, "controlled forward failure"):
                    with fixed_generation(current):
                        model.generate(input_ids=torch.tensor([[3, 6]]), generation_config=config)
            finally:
                failure_hook.remove()
            self.assertIsNone(config.min_new_tokens)
            natural = generate(
                current, [{"role": "user", "content": "answer"}],
                max_new_tokens=3, temperature=0, top_p=1,
            )
            self.assertEqual(natural.generated_ids, [2])
            self.assertEqual(natural.finish_reason, "stop")
            self.assertEqual(model.generation_config.to_dict(), defaults)
            self.assertEqual(model.generate, original_generate)
        finally:
            hook.remove()

    def test_shared_model_config_cannot_be_mutated(self):
        current = runtime()
        before = current.model.generation_config.to_dict()
        with self.assertRaisesRegex(ValueError, "fresh per-call"):
            with fixed_generation(current):
                current.model.generate(
                    input_ids=torch.tensor([[3, 6]]), generation_config=current.model.generation_config,
                )
        self.assertEqual(current.model.generation_config.to_dict(), before)


class BaselineServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_fixed_and_natural_phases_keep_defaults_and_stop_server(self):
        current = runtime()
        defaults = current.model.generation_config.to_dict()
        hook = current.model.register_forward_hook(prefer_eos)
        record = new_backend_result("transformers")
        try:
            await exercise_baseline(current, record, port=0, timeout=60)
        finally:
            hook.remove()
        self.assertEqual(record["status"], "completed")
        self.assertTrue(record["checks"]["all_passed"])
        self.assertTrue(record["checks"]["warmup_input_ids_match"])
        self.assertTrue(record["checks"]["server_stopped_cleanly"])
        self.assertEqual(current.model.generation_config.to_dict(), defaults)
        self.assertIsNone(api.app.state.runtime)
        self.assertIsNone(api.app.state.executor)
        rows = [row for scenario in record["scenarios"] for row in scenario["rows"]]
        self.assertEqual([row["completion_tokens"] for row in rows], [16, 128, 32] * 4 + [1, 1])
        self.assertTrue(all(row["prompt_token_ids"] is None for row in rows))
        for observation in record["generation_observations"]:
            controls = observation["generation_controls"]
            expected = None if "natural_eos" in observation["scenario"] else controls["max_new_tokens"]
            self.assertEqual(controls["min_new_tokens"], expected)
        async with httpx.AsyncClient(trust_env=False) as client:
            with self.assertRaises(httpx.ConnectError):
                await client.get(record["runtime"]["url"] + "/ready")

    async def test_generation_failure_retains_partial_evidence_and_restores_worker(self):
        current = runtime()
        original_generate = current.model.generate
        defaults = current.model.generation_config.to_dict()
        prefills = 0

        def fail_second_request(model, args, kwargs):
            nonlocal prefills
            ids = kwargs.get("input_ids")
            if ids is not None and ids.shape[1] > 1:
                prefills += 1
                if prefills == 2:
                    raise RuntimeError("controlled second request failure")

        failure_hook = current.model.register_forward_pre_hook(fail_second_request, with_kwargs=True)
        logits_hook = current.model.register_forward_hook(prefer_eos)
        record = new_backend_result("transformers")
        try:
            with self.assertRaisesRegex(RuntimeError, "generation_failed"):
                await exercise_baseline(current, record, port=0, timeout=60)
        finally:
            failure_hook.remove()
            logits_hook.remove()
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["checks"]["all_passed"])
        self.assertTrue(record["checks"]["server_stopped_cleanly"])
        rows = record["scenarios"][0]["rows"]
        self.assertEqual([row["status"] for row in rows], ["success", "error"])
        self.assertEqual(rows[0]["completion_tokens"], 16)
        self.assertEqual(rows[1]["error_code"], "generation_failed")
        self.assertEqual(current.model.generation_config.to_dict(), defaults)
        self.assertEqual(current.model.generate, original_generate)
        self.assertIsNone(api.app.state.executor)
        self.assertFalse(api.app.state.producer_futures)

    async def test_canceling_http_measurement_cancels_and_drains_generation(self):
        current = runtime()
        entered, release = Event(), Event()

        def block_forward(model, args):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("Test failed to release the model worker.")

        hook = current.model.register_forward_pre_hook(block_forward)
        record = new_backend_result("transformers")
        task = asyncio.create_task(exercise_baseline(current, record, port=0, timeout=30))
        try:
            async with asyncio.timeout(10):
                while not entered.is_set():
                    if task.done():
                        await task
                    await asyncio.sleep(0.01)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            hook.remove()
        self.assertEqual(record["status"], "failed")
        self.assertFalse(record["checks"]["all_passed"])
        self.assertTrue(record["checks"]["server_stopped_cleanly"])
        self.assertEqual(record["scenarios"][0]["rows"][0]["status"], "canceled")
        self.assertIsNone(api.app.state.executor)
        self.assertFalse(api.app.state.producer_futures)


class BaselineCliTests(unittest.TestCase):
    def test_cuda_failure_persists_evidence_and_never_overwrites_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline.json"
            arguments = [
                "engine_comparison", "--baseline", "--output", str(output),
                "--revision", "a" * 40, "--timeout", "30",
            ]
            with patch("sys.argv", arguments), patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(main(), 1)
                record = json.loads(output.read_text())
                self.assertEqual(record["status"], "failed")
                self.assertIn("no CPU fallback", record["error"])
                self.assertFalse(record["checks"]["all_passed"])
                before = output.read_bytes()
                with self.assertRaises(FileExistsError):
                    main()
                self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
