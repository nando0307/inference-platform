"""CPU-only token callbacks and scoped instrumentation for benchmark commands."""

from contextlib import contextmanager
from time import perf_counter
from unittest.mock import patch

from benchmarks.common import peak_rss_bytes


class TokenTimingError(ValueError):
    """The model callbacks do not describe one generated token per step."""


class TokenRecorder:
    """Keep timestamps only; forward the original tensors without decoding them."""

    def __init__(self, downstream=None, *, clock=perf_counter):
        self.downstream = downstream
        self.clock = clock
        self.started = None
        self.prompt_tokens = None
        self.timestamps = []

    def start(self):
        self.started = self.clock()

    def put(self, value):
        if self.prompt_tokens is None:
            if len(value.shape) != 2 or value.shape[0] != 1:
                raise TokenTimingError("Expected one full prompt in the first callback.")
            self.prompt_tokens = value.shape[1]
        else:
            timestamp = self.clock()
            if value.numel() != 1:
                raise TokenTimingError("Expected one generated token per callback.")
            self.timestamps.append(timestamp)
        if self.downstream is not None:
            self.downstream.put(value)

    def end(self):
        if self.downstream is not None:
            self.downstream.end()

    def verify(self, prompt_tokens, completion_tokens):
        if self.prompt_tokens != prompt_tokens:
            raise TokenTimingError("Prompt callback count differs from model input.")
        if len(self.timestamps) != completion_tokens:
            raise TokenTimingError("Token callback count differs from generated IDs.")

    def metrics(self, finished):
        intervals = [
            (later - earlier) * 1000
            for earlier, later in zip(self.timestamps, self.timestamps[1:])
        ]
        return {
            "ttft_ms": (
                (self.timestamps[0] - self.started) * 1000
                if self.timestamps else None
            ),
            "tpot_ms": (
                (self.timestamps[-1] - self.timestamps[0]) * 1000
                / (len(self.timestamps) - 1)
                if len(self.timestamps) > 1 else None
            ),
            "inter_token_ms": intervals,
            "generation_ms": (finished - self.started) * 1000,
            "observed_token_callbacks": len(self.timestamps),
        }


def load_cached_runtime():
    """Reuse production loading, but pin both artifacts and prohibit downloads."""
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from src import inference

    try:
        snapshot = snapshot_download(inference.MODEL_ID, local_files_only=True)
        revision = Path(snapshot).name
        tokenizer_loader = inference.AutoTokenizer.from_pretrained
        model_loader = inference.AutoModelForCausalLM.from_pretrained

        def cached_tokenizer(*args, **kwargs):
            return tokenizer_loader(
                *args, **{**kwargs, "local_files_only": True, "revision": revision}
            )

        def cached_model(*args, **kwargs):
            return model_loader(
                *args, **{**kwargs, "local_files_only": True, "revision": revision}
            )

        with (
            patch.object(inference.AutoTokenizer, "from_pretrained", cached_tokenizer),
            patch.object(inference.AutoModelForCausalLM, "from_pretrained", cached_model),
        ):
            runtime = inference.load_runtime()
        runtime.benchmark_model_revision = revision
        return runtime
    except OSError:
        raise RuntimeError(
            "Benchmark loading requires the complete Qwen model and tokenizer in the "
            "local Hugging Face cache. Download them separately before benchmarking."
        ) from None


@contextmanager
def instrument_generation(runtime, emit, *, fixed_length=False):
    """Patch this runtime only; emit after stopping the model-boundary timer."""
    import torch

    if runtime.model.device.type != "cpu" or runtime.model.dtype != torch.float32:
        raise ValueError("These benchmarks require the CPU float32 baseline.")
    original_generate = runtime.model.generate

    def measured_generate(*args, **kwargs):
        input_ids = kwargs["input_ids"]
        config = kwargs["generation_config"]
        prompt_tokens = input_ids.shape[1]
        cap = config.max_new_tokens
        if input_ids.shape[0] != 1 or not 1 <= cap <= 512:
            raise ValueError("Benchmark generation requires batch one and a 1..512 output cap.")
        if prompt_tokens + cap > runtime.context_limit:
            raise ValueError("Benchmark generation exceeds the local context budget.")
        if fixed_length:
            if config is runtime.model.generation_config:
                raise ValueError("Fixed-length generation requires a fresh per-call config.")
            config.min_new_tokens = cap
        settings = {
            name: getattr(config, name)
            for name in (
                "max_new_tokens", "min_new_tokens", "use_cache", "num_beams",
                "num_return_sequences", "repetition_penalty", "do_sample",
                "temperature", "top_p", "top_k",
            )
        }
        recorder = TokenRecorder(kwargs.get("streamer"))
        kwargs["streamer"] = recorder
        row = {
            "status": "error",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": None,
            "max_new_tokens": cap,
            "generation_settings": settings,
        }
        finished = None
        recorder.start()
        try:
            output = original_generate(*args, **kwargs)
            finished = perf_counter()
            count = output.shape[1] - prompt_tokens
            recorder.verify(prompt_tokens, count)
            if fixed_length and count != cap:
                raise TokenTimingError("Fixed-length generation stopped before its output cap.")
            row.update(status="success", completion_tokens=count)
            return output
        except BaseException as error:
            if finished is None:
                finished = perf_counter()
            row.update(error_code="generation_failed", exception_type=type(error).__name__)
            raise
        finally:
            row.update(recorder.metrics(finished))
            row["process_peak_rss_bytes"] = peak_rss_bytes()
            emit(row)

    with patch.object(runtime.model, "generate", measured_generate):
        yield
