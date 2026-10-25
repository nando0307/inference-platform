import unittest
from queue import Empty
from threading import Event, Thread
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from src.inference import CancellationToken, GenerationCancelled
from src.streaming import BoundedTextIteratorStreamer


class BoundedStreamingTests(unittest.TestCase):
    def setUp(self):
        backend = Tokenizer(WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
             "Olá": 3, "café!": 4, "你好": 5, "世界！": 6},
            unk_token="[UNK]",
        ))
        backend.pre_tokenizer = WhitespaceSplit()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
            eos_token="[EOS]",
        )

    def streamer(self, **kwargs):
        return BoundedTextIteratorStreamer(
            self.tokenizer, cancellation=CancellationToken(), skip_prompt=True,
            skip_special_tokens=True, clean_up_tokenization_spaces=False,
            timeout=0.05, **kwargs,
        )

    def test_bounded_decode_preserves_multilingual_text_and_invisible_eos(self):
        streamer = self.streamer()
        ids = [3, 4, 5, 6, 2]
        streamer.put(torch.tensor([[0, 1]]))
        for token in ids:
            streamer.put(torch.tensor([token]))
        streamer.end()
        text = "".join(streamer)
        self.assertEqual(text, self.tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))
        self.assertNotIn("[EOS]", text)
        self.assertNotIn("[UNK]", text)

    def test_empty_increments_do_not_exhaust_capacity_before_terminal(self):
        streamer = self.streamer(maxsize=2)
        for _ in range(20):
            streamer.on_finalized_text("")
        streamer.on_finalized_text("visible")
        streamer.on_finalized_text("", stream_end=True)
        self.assertEqual(list(streamer), ["visible"])

    def test_stalled_full_buffer_cancels_instead_of_growing_or_hanging(self):
        streamer = self.streamer(maxsize=1, stall_timeout=0.05)
        streamer.on_finalized_text("first")
        with self.assertRaises(GenerationCancelled) as caught:
            streamer.on_finalized_text("second")
        self.assertEqual(caught.exception.reason, "slow_consumer")
        self.assertEqual(streamer.text_queue.qsize(), 1)
        self.assertTrue(streamer.cancellation.is_cancelled())
        # Even the terminal sentinel must not block after cancellation.
        with self.assertRaises(GenerationCancelled):
            streamer.on_finalized_text("", stream_end=True)

    def test_disconnect_interrupts_a_blocked_put_before_stall_deadline(self):
        streamer = self.streamer(maxsize=1, stall_timeout=10)
        streamer.on_finalized_text("first")
        entered = Event()
        failures = []
        original_put = streamer.text_queue.put

        def observed_put(*args, **kwargs):
            entered.set()
            return original_put(*args, **kwargs)

        def produce():
            try:
                streamer.on_finalized_text("second")
            except BaseException as error:
                failures.append(error)

        worker = Thread(target=produce)
        with patch.object(streamer.text_queue, "put", observed_put):
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                streamer.cancellation.cancel("client_disconnected")
                worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(len(failures), 1)
                self.assertIsInstance(failures[0], GenerationCancelled)
                self.assertEqual(failures[0].reason, "client_disconnected")
            finally:
                streamer.cancellation.cancel("request_canceled")
                try:
                    streamer.text_queue.get_nowait()
                except Empty:
                    pass
                worker.join(2)

    def test_unbounded_capacity_and_nonfinite_stall_deadlines_are_rejected(self):
        for options in (
            {"maxsize": 0}, {"maxsize": -1}, {"maxsize": 1.5},
            {"stall_timeout": 0}, {"stall_timeout": float("inf")},
            {"stall_timeout": float("nan")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.streamer(**options)


if __name__ == "__main__":
    unittest.main()
