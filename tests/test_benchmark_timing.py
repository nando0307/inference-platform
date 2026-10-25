import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast, TextIteratorStreamer

from benchmarks.timing import TokenRecorder, TokenTimingError, instrument_generation
from src.inference import CancellationToken, ContextLimitError, GenerationCancelled, Runtime, generate


def tiny_tokenizer():
    backend = Tokenizer(WordLevel(
        {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "[user]": 3,
         "answer": 4, "tail": 5, "[assistant]": 6},
        unk_token="[UNK]",
    ))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '[' + message['role'] + ']' }} "
        "{{ message['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}[assistant]{% endif %}"
    )
    return tokenizer


class TokenRecorderTests(unittest.TestCase):
    def test_prompt_exclusion_token_timestamps_and_invisible_eos(self):
        streamer = TextIteratorStreamer(tiny_tokenizer(), skip_prompt=True,
                                        skip_special_tokens=True, timeout=0.1)
        clock = iter([10.0, 10.02, 10.05, 10.11])
        recorder = TokenRecorder(streamer, clock=lambda: next(clock))
        recorder.start()
        recorder.put(torch.tensor([[3, 4, 6]]))
        recorder.put(torch.tensor([4]))
        # A token is already timed while the actual text streamer still buffers it.
        self.assertEqual(next(streamer), "")
        self.assertAlmostEqual(recorder.metrics(10.03)["ttft_ms"], 20)
        recorder.put(torch.tensor([5]))
        self.assertEqual(next(streamer), "answer ")
        recorder.put(torch.tensor([2]))
        self.assertEqual(next(streamer), "")
        recorder.end()
        self.assertEqual(list(streamer), ["tail"])
        recorder.verify(3, 3)
        metrics = recorder.metrics(10.2)
        self.assertEqual(metrics["observed_token_callbacks"], 3)
        self.assertAlmostEqual(metrics["ttft_ms"], 20)
        self.assertAlmostEqual(metrics["tpot_ms"], 45)
        self.assertAlmostEqual(metrics["inter_token_ms"][0], 30)
        self.assertAlmostEqual(metrics["inter_token_ms"][1], 60)
        self.assertAlmostEqual(metrics["generation_ms"], 200)

    def test_zero_and_one_output_have_no_decode_interval(self):
        for count in (0, 1):
            with self.subTest(count=count):
                clock = iter([1.0, 1.25])
                recorder = TokenRecorder(clock=lambda: next(clock))
                recorder.start()
                recorder.put(torch.tensor([[3, 6]]))
                if count:
                    recorder.put(torch.tensor([2]))
                recorder.end()
                recorder.verify(2, count)
                metrics = recorder.metrics(1.5)
                self.assertIsNone(metrics["tpot_ms"])
                self.assertEqual(metrics["inter_token_ms"], [])
                self.assertEqual(metrics["ttft_ms"], 250 if count else None)
                self.assertEqual(metrics["generation_ms"], 500)

    def test_callback_count_mismatch_cannot_be_reported_as_success(self):
        recorder = TokenRecorder()
        recorder.start()
        recorder.put(torch.tensor([[3, 6]]))
        recorder.put(torch.tensor([2]))
        with self.assertRaises(TokenTimingError):
            recorder.verify(2, 2)
        with self.assertRaises(TokenTimingError):
            recorder.verify(3, 1)
        with self.assertRaises(TokenTimingError):
            recorder.put(torch.tensor([4, 5]))


class InstrumentedGenerationTests(unittest.TestCase):
    def setUp(self):
        tokenizer = tiny_tokenizer()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=len(tokenizer), n_positions=16, n_embd=8, n_layer=1,
                n_head=1, bos_token_id=None, eos_token_id=2, pad_token_id=0,
            ))
        model.eval()
        self.runtime = Runtime(model, tokenizer, {2}, context_limit=6)
        self.messages = [{"role": "user", "content": "answer"}]

    def generate(self, cap):
        return generate(self.runtime, self.messages, max_new_tokens=cap, temperature=0, top_p=1)

    def test_fixed_length_respects_budget_without_leaking_into_natural_generation(self):
        def prefer_eos(model, args, output):
            output.logits.fill_(-10)
            output.logits[..., 4] = 5
            output.logits[..., 2] = 10

        model = self.runtime.model
        defaults = model.generation_config.to_dict()
        hook = model.register_forward_hook(prefer_eos)
        rows = []
        try:
            with instrument_generation(self.runtime, rows.append):
                natural = self.generate(1)
            with instrument_generation(self.runtime, rows.append, fixed_length=True):
                fixed = self.generate(3)
                with self.assertRaises(ContextLimitError):
                    self.generate(4)
            natural_after = self.generate(3)
        finally:
            hook.remove()
        self.assertEqual(natural.generated_ids, [2])
        self.assertEqual(natural.finish_reason, "stop")
        self.assertEqual(fixed.generated_ids, [4, 4, 4])
        self.assertEqual(fixed.finish_reason, "length")
        self.assertEqual(fixed.prompt_tokens + len(fixed.generated_ids), 6)
        self.assertEqual(natural_after.generated_ids, [2])
        self.assertEqual(model.generation_config.to_dict(), defaults)
        self.assertEqual([row["completion_tokens"] for row in rows], [1, 3])
        self.assertIsNone(rows[0]["tpot_ms"])
        self.assertGreaterEqual(rows[1]["tpot_ms"], 0)
        self.assertEqual(rows[1]["generation_settings"]["min_new_tokens"], 3)

    def test_model_failure_propagates_without_fabricating_completion_counts(self):
        failure = RuntimeError("PRIVATE-MARKER")

        def fail_forward(model, args):
            raise failure

        hook = self.runtime.model.register_forward_pre_hook(fail_forward)
        rows = []
        try:
            with self.assertRaises(RuntimeError) as caught:
                with instrument_generation(self.runtime, rows.append):
                    self.generate(2)
        finally:
            hook.remove()
        self.assertIs(caught.exception, failure)
        self.assertEqual(rows[0]["status"], "error")
        self.assertIsNone(rows[0]["completion_tokens"])
        self.assertIsNone(rows[0]["ttft_ms"])
        self.assertEqual(rows[0]["exception_type"], "RuntimeError")
        self.assertNotIn("PRIVATE-MARKER", str(rows[0]))
        with instrument_generation(self.runtime, rows.append, fixed_length=True):
            result = self.generate(2)
        self.assertEqual(len(result.generated_ids), 2)
        self.assertEqual(rows[-1]["status"], "success")
        self.assertEqual(rows[-1]["completion_tokens"], 2)

    def test_cooperative_stopping_and_streamer_cancellation_do_not_fabricate_completion_usage(self):
        for raise_from_streamer in (False, True):
            with self.subTest(raise_from_streamer=raise_from_streamer):
                token = CancellationToken()

                class CancelAfterToken:
                    def __init__(self):
                        self.prompt_seen = False

                    def put(self, value):
                        if not self.prompt_seen:
                            self.prompt_seen = True
                            return
                        token.cancel("client_disconnected")
                        if raise_from_streamer:
                            token.raise_if_cancelled()

                    def end(self):
                        pass

                rows = []
                with self.assertRaises(GenerationCancelled):
                    with instrument_generation(
                        self.runtime, rows.append, get_cancellation=lambda: token,
                    ):
                        generate(
                            self.runtime, self.messages, max_new_tokens=3, temperature=0, top_p=1,
                            streamer=CancelAfterToken(), cancellation=token,
                        )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["status"], "canceled")
                self.assertEqual(rows[0]["cancellation_reason"], "client_disconnected")
                self.assertIsNone(rows[0]["completion_tokens"])
                self.assertGreaterEqual(rows[0]["ttft_ms"], 0)
                self.assertNotIn("exception_type", rows[0])


if __name__ == "__main__":
    unittest.main()
