import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from src.inference import (
    CancellationToken, ContextLimitError, GenerationCancelled, Runtime,
    generate, load_runtime,
)


class ContextBudgetTests(unittest.TestCase):
    def setUp(self):
        # An in-memory tokenizer and tiny model keep boundary tests offline.
        vocabulary = {
            token: index
            for index, token in enumerate(
                [
                    "[UNK]",
                    "[PAD]",
                    "[EOS]",
                    "[BOS]",
                    "[system]",
                    "[user]",
                    "[assistant]",
                    "older",
                    "reply",
                    "latest",
                ]
            )
        }
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = WhitespaceSplit()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            eos_token="[EOS]",
            bos_token="[BOS]",
            model_max_length=128,
        )
        tokenizer.chat_template = (
            "{{ bos_token }} "
            "{% for message in messages %}"
            "{{ '[' + message['role'] + ']' }} {{ message['content'] }} {{ eos_token }} "
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '[assistant]' }}{% endif %}"
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=len(vocabulary),
                    n_positions=64,
                    n_embd=8,
                    n_layer=1,
                    n_head=1,
                    bos_token_id=None,
                    eos_token_id=None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            )
        model.eval()
        self.runtime = Runtime(model, tokenizer, set(), context_limit=19)
        self.messages = [
            {"role": "system", "content": "older"},
            {"role": "user", "content": "older older"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "latest"},
        ]
        # Five content tokens, eight turn markers, BOS, and assistant prefix.
        self.prompt_tokens = 15

    def run_generation(self, messages=None, max_new_tokens=4, on_prepared=None, cancellation=None):
        return generate(
            self.runtime,
            self.messages if messages is None else messages,
            max_new_tokens=max_new_tokens,
            temperature=0,
            top_p=1,
            on_prepared=on_prepared,
            cancellation=cancellation,
        )

    def test_exact_fit_keeps_full_history_and_requested_output_budget(self):
        result = self.run_generation()
        self.assertEqual(result.prompt_tokens, self.prompt_tokens)
        self.assertEqual(len(result.generated_ids), 4)
        self.assertEqual(result.prompt_tokens + len(result.generated_ids), 19)
        self.assertEqual(result.finish_reason, "length")

    def test_either_eos_at_output_cap_reports_stop_and_counts_invisible_token(self):
        tokenizer = self.runtime.tokenizer
        eos_ids = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        self.runtime.model.generation_config.eos_token_id = eos_ids
        self.runtime.eos_token_ids = set(eos_ids)
        visible_id = tokenizer.convert_tokens_to_ids("reply")
        for eos_id in eos_ids:
            with self.subTest(eos_id=eos_id):
                selected_ids = iter([visible_id, eos_id])

                def select_token(model, args, output):
                    # Control logits, not generate(): exercise real stopping/decoding.
                    output.logits.fill_(float("-inf"))
                    output.logits[..., next(selected_ids)] = 0

                hook = self.runtime.model.register_forward_hook(select_token)
                try:
                    result = self.run_generation(max_new_tokens=2)
                finally:
                    hook.remove()
                self.assertEqual(result.generated_ids, [visible_id, eos_id])
                self.assertEqual(len(result.generated_ids), 2)
                self.assertEqual(result.finish_reason, "stop")
                self.assertEqual(result.text, "reply")
                self.assertEqual(result.prompt_tokens, self.prompt_tokens)

    def test_one_token_over_rejects_before_preparation_and_generation(self):
        prepared = Mock()
        self.messages[-1]["content"] = "PRIVATE-CONTEXT-MARKER"
        with patch.object(self.runtime.model, "generate") as model_generate:
            with self.assertRaises(ContextLimitError) as caught:
                self.run_generation(max_new_tokens=5, on_prepared=prepared)
        prepared.assert_not_called()
        model_generate.assert_not_called()
        message = str(caught.exception)
        for count in (self.prompt_tokens, 5, self.runtime.context_limit):
            self.assertIn(str(count), message)
        self.assertNotIn("PRIVATE-CONTEXT-MARKER", message)

    def test_oversized_prompt_is_not_silently_truncated(self):
        self.runtime.context_limit = 64
        messages = [{"role": "user", "content": "older " * 80}]
        prepared = Mock()
        with patch.object(self.runtime.model, "generate") as model_generate:
            with self.assertRaises(ContextLimitError):
                self.run_generation(messages, max_new_tokens=1, on_prepared=prepared)
        prepared.assert_not_called()
        model_generate.assert_not_called()

    def test_earlier_turns_cannot_be_discarded_to_fit(self):
        self.runtime.context_limit = 9
        latest_only = self.run_generation(self.messages[-1:])
        self.assertEqual(latest_only.prompt_tokens, 5)
        self.assertEqual(len(latest_only.generated_ids), 4)
        with self.assertRaises(ContextLimitError):
            self.run_generation()

    def test_preparation_precedes_model_execution_for_valid_request(self):
        events = []
        hook = self.runtime.model.register_forward_pre_hook(
            lambda model, args: events.append("forward")
        )
        try:
            self.run_generation(on_prepared=lambda: events.append("prepared"))
        finally:
            hook.remove()
        self.assertEqual(events[0:2], ["prepared", "forward"])
        self.assertEqual(events.count("prepared"), 1)

    def test_cancellation_before_or_after_preparation_never_enters_model(self):
        for during_preparation in (False, True):
            with self.subTest(during_preparation=during_preparation):
                token = CancellationToken()
                forwards = []
                hook = self.runtime.model.register_forward_pre_hook(
                    lambda model, args: forwards.append(True)
                )
                if not during_preparation:
                    token.cancel("client_disconnected")
                try:
                    with self.assertRaises(GenerationCancelled) as caught:
                        self.run_generation(
                            cancellation=token,
                            on_prepared=lambda: token.cancel("client_disconnected"),
                        )
                finally:
                    hook.remove()
                self.assertEqual(caught.exception.reason, "client_disconnected")
                self.assertEqual(forwards, [])

    def test_running_cancellation_stops_after_current_step_and_is_request_local(self):
        token = CancellationToken()
        forwards = []

        def cancel_after_forward(model, args, output):
            forwards.append(True)
            token.cancel("client_disconnected")

        hook = self.runtime.model.register_forward_hook(cancel_after_forward)
        try:
            with self.assertRaises(GenerationCancelled):
                self.run_generation(cancellation=token)
        finally:
            hook.remove()
        # Post-generation cancellation alone would waste all four forward calls.
        self.assertEqual(len(forwards), 1)
        recovered = self.run_generation()
        self.assertEqual(len(recovered.generated_ids), 4)
        self.assertEqual(recovered.finish_reason, "length")

    def test_first_cancellation_reason_and_timestamp_are_retained(self):
        token = CancellationToken()
        token.cancel("client_disconnected")
        first_requested_at = token.requested_at
        token.cancel("shutdown")
        self.assertEqual(token.reason, "client_disconnected")
        self.assertIsNotNone(first_requested_at)
        self.assertEqual(token.requested_at, first_requested_at)
        with self.assertRaises(GenerationCancelled) as caught:
            token.raise_if_cancelled()
        self.assertEqual(caught.exception.reason, "client_disconnected")


class RuntimeContextLimitTests(unittest.TestCase):
    def load_with_config(self, config):
        model = Mock()
        model.to.return_value = model
        model.config = config
        model.generation_config.eos_token_id = [2]
        with (
            patch(
                "src.inference.AutoModelForCausalLM.from_pretrained", return_value=model
            ),
            patch(
                "src.inference.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(pad_token_id=1),
            ),
        ):
            return load_runtime()

    def test_supported_context_respects_smaller_model_and_local_cap(self):
        for declared, expected in ((1024, 1024), (32768, 2048)):
            with self.subTest(declared=declared):
                runtime = self.load_with_config(
                    SimpleNamespace(max_position_embeddings=declared)
                )
                self.assertEqual(runtime.context_limit, expected)

    def test_absent_or_invalid_declared_context_fails_loading(self):
        configs = [SimpleNamespace()]
        configs.extend(
            SimpleNamespace(max_position_embeddings=value)
            for value in (None, True, False, 0, -1, 2048.0, "2048")
        )
        for config in configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    self.load_with_config(config)


if __name__ == "__main__":
    unittest.main()
