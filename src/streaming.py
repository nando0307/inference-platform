"""Bound text buffering without replacing the model tokenizer's incremental decode."""

from math import isfinite
from queue import Full, Queue
from time import perf_counter

from transformers import TextIteratorStreamer

from .inference import CancellationToken


class BoundedTextIteratorStreamer(TextIteratorStreamer):
    def __init__(
        self, tokenizer, *, cancellation: CancellationToken,
        maxsize: int = 8, stall_timeout: float = 1.0, **kwargs,
    ):
        if type(maxsize) is not int or maxsize < 1:
            raise ValueError("The streaming queue capacity must be a positive integer.")
        if not isfinite(stall_timeout) or stall_timeout <= 0:
            raise ValueError("The streaming stall timeout must be finite and positive.")
        super().__init__(tokenizer, **kwargs)
        self.cancellation = cancellation
        self.text_queue = Queue(maxsize=maxsize)
        self.stall_timeout = stall_timeout

    def put(self, value):
        self.cancellation.raise_if_cancelled()
        super().put(value)

    def end(self):
        self.cancellation.raise_if_cancelled()
        super().end()

    def _put_text(self, text):
        deadline = perf_counter() + self.stall_timeout
        while True:
            self.cancellation.raise_if_cancelled()
            remaining = deadline - perf_counter()
            if remaining <= 0:
                self.cancellation.cancel("slow_consumer")
                self.cancellation.raise_if_cancelled()
            try:
                self.text_queue.put(text, timeout=min(0.05, remaining))
                return
            except Full:
                continue

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.cancellation.raise_if_cancelled()
        # Empty decode increments consume no buffer capacity, but EOS still ends it.
        if text:
            self._put_text(text)
        if stream_end:
            self._put_text(self.stop_signal)
