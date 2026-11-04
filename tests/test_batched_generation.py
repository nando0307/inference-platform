import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.batched_generation import generate_batch, main
from src.inference import ContextLimitError, Runtime, generate
from test_benchmark_timing import tiny_tokenizer


class BatchedGenerationTests(unittest.TestCase):
    def setUp(self):
        tokenizer = tiny_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        # Both configured EOS IDs are special tokens, as in the Qwen tokenizer.
        tokenizer.add_special_tokens({"additional_special_tokens": ["[assistant]"]})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=64, n_embd=8, n_layer=1,
                n_head=1, bos_token_id=None, eos_token_id=[2, 6], pad_token_id=2,
            )).eval()
        self.runtime = Runtime(model, tokenizer, {2, 6}, context_limit=64)
        self.conversations = [
            [{"role": "user", "content": "answer"}],
            [{"role": "user", "content": "tail tail tail"}],
        ]

    def test_mixed_lengths_independent_eos_and_output_cap(self):
        # Real HF generation and cache progression, with deterministic next-token
        # logits to exercise unequal stopping times independently of model quality.
        step = 0
        short_rows = None

        def choose_tokens(model, args, kwargs, output):
            nonlocal step, short_rows
            ids = kwargs["input_ids"]
            if ids.shape[1] > 1:
                step = 0
                short_rows = (ids == 4).any(dim=1).tolist()
            output.logits.fill_(-20)
            for row, short in enumerate(short_rows):
                token_id = 2 if short else 6 if step == 3 else 5
                output.logits[row, :, token_id] = 20
            step += 1

        hook = self.runtime.model.register_forward_hook(choose_tokens, with_kwargs=True)
        try:
            for cap, second_ids, finish in ((3, [5, 5, 5], "length"), (4, [5, 5, 5, 6], "stop")):
                with self.subTest(cap=cap):
                    results = generate_batch(self.runtime, self.conversations, max_new_tokens=cap)
                    self.assertEqual([row.prompt_tokens for row in results], [3, 5])
                    self.assertEqual(results[0].generated_ids, [2])
                    self.assertEqual(results[0].finish_reason, "stop")
                    self.assertEqual(results[0].text, "")
                    self.assertEqual(results[1].generated_ids, second_ids)
                    self.assertEqual(results[1].finish_reason, finish)
                    self.assertEqual(results[1].text, "tail tail tail")
            reversed_results = generate_batch(self.runtime, self.conversations[::-1], max_new_tokens=4)
            self.assertEqual([row.generated_ids for row in reversed_results], [[5, 5, 5, 6], [2]])
        finally:
            hook.remove()

    def test_real_greedy_batch_matches_single_and_preserves_later_calls(self):
        self.runtime.model.generation_config.eos_token_id = 2
        self.runtime.eos_token_ids = {2}
        tokenizer = self.runtime.tokenizer
        padding_before = tokenizer(["answer", "tail tail"], padding=True)["attention_mask"]
        singles = [
            generate(self.runtime, messages, max_new_tokens=4, temperature=0, top_p=1)
            for messages in self.conversations
        ]
        batched = generate_batch(self.runtime, self.conversations, max_new_tokens=4)
        self.assertEqual(batched, singles)
        self.assertEqual(
            tokenizer(["answer", "tail tail"], padding=True)["attention_mask"], padding_before,
        )
        self.assertEqual(
            generate(self.runtime, self.conversations[0], max_new_tokens=4, temperature=0, top_p=1),
            singles[0],
        )

    def test_context_boundary_and_invalid_inputs_do_not_truncate(self):
        self.runtime.context_limit = 8
        with patch.object(self.runtime.model, "forward", side_effect=AssertionError("Model must not run")):
            with self.assertRaises(ContextLimitError):
                generate_batch(self.runtime, self.conversations, max_new_tokens=4)
            for conversations, cap in (([], 3), ([[]], 3), (self.conversations, 0),
                                       (self.conversations, True), (self.conversations, 513)):
                with self.subTest(conversations=conversations, cap=cap):
                    with self.assertRaises(ValueError):
                        generate_batch(self.runtime, conversations, max_new_tokens=cap)
        # Five padded input positions plus three outputs exactly fit the budget.
        results = generate_batch(self.runtime, self.conversations, max_new_tokens=3)
        self.assertEqual([row.prompt_tokens for row in results], [3, 5])
        self.assertTrue(all(1 <= len(row.generated_ids) <= 3 for row in results))

    def test_failure_preserves_completed_reference_and_refuses_overwrite(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        with (
            TemporaryDirectory() as directory,
            patch.dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false"),
            patch("src.batched_generation.load_runtime", return_value=self.runtime),
            patch("src.batched_generation.generate_batch", side_effect=RuntimeError("private detail")),
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
        ):
            output = Path(directory) / "result.json"
            args = ["--output", str(output), "--max-new-tokens", "3"]
            self.assertEqual(main(args), 1)
            saved = output.read_bytes()
            result = json.loads(saved)
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["exception_type"], "RuntimeError")
            self.assertEqual(len(result["sequential"]["results"]), 2)
            self.assertEqual(result["batched"]["results"], [])
            self.assertIsNone(result["all_token_ids_equal"])
            self.assertNotIn(b"private detail", saved)
            with self.assertRaises(SystemExit) as raised:
                main(args)
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
