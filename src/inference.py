from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from time import perf_counter
from typing import Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Runtime:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    eos_token_ids: set[int]
    context_limit: int


device = "cpu"


def load_runtime(*, revision: str | None = None) -> Runtime:
    """Load shared inference resources; the caller owns their lifetime."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        dtype=torch.float32,
    ).to(device)
    model.eval()

    configured_eos = model.generation_config.eos_token_id
    if configured_eos is None:
        eos_token_ids = set()
    elif isinstance(configured_eos, int):
        eos_token_ids = {configured_eos}
    else:
        eos_token_ids = set(configured_eos)

    if tokenizer.pad_token_id is None and not eos_token_ids:
        raise ValueError(
            "Model requires a tokenizer pad token or a configured EOS token."
        )

    declared_limit = getattr(model.config, "max_position_embeddings", None)
    if (
        not isinstance(declared_limit, int)
        or isinstance(declared_limit, bool)
        or declared_limit <= 0
    ):
        raise ValueError("Model max_position_embeddings must be a positive integer.")

    return Runtime(
        model=model,
        tokenizer=tokenizer,
        eos_token_ids=eos_token_ids,
        context_limit=min(2048, declared_limit),
    )


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_ids: list[int]
    finish_reason: Literal["stop", "length"]


class ContextLimitError(ValueError):
    """The formatted prompt and requested output exceed the supported context."""


class GenerationCancelled(RuntimeError):
    """Cooperative cancellation is not a successful truncated completion."""

    def __init__(self, reason: str):
        super().__init__("Generation was canceled.")
        self.reason = reason


class CancellationToken:
    """First cancellation reason/time wins across the HTTP and producer threads."""

    def __init__(self):
        self._event = Event()
        self._lock = Lock()
        self.reason: str | None = None
        self.requested_at: float | None = None

    def cancel(self, reason: str = "request_canceled") -> None:
        with self._lock:
            if not self._event.is_set():
                self.reason = reason
                self.requested_at = perf_counter()
                self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            assert self.reason is not None
            raise GenerationCancelled(self.reason)


class _CancellationCriteria(StoppingCriteria):
    def __init__(self, cancellation: CancellationToken):
        self.cancellation = cancellation

    def __call__(self, input_ids, scores, **kwargs):
        # A scalar boolean broadcasts across the single supported sequence.
        return self.cancellation.is_cancelled()


@torch.inference_mode()
def generate(
    runtime: Runtime,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    streamer: TextIteratorStreamer | None = None,
    on_prepared: Callable[[], None] | None = None,
    cancellation: CancellationToken | None = None,
) -> GenerationResult:
    """Generate one completion using caller-owned shared model resources."""
    if temperature == 0 and top_p != 1.0:
        raise ValueError("Greedy decoding requires top_p=1.0.")
    if cancellation is not None:
        cancellation.raise_if_cancelled()

    text = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = runtime.tokenizer(
        [text],
        return_tensors="pt",
        add_special_tokens=False,
        return_attention_mask=True,
        truncation=False,
    ).to(runtime.model.device)
    prompt_tokens = model_inputs["input_ids"].shape[1]
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    if prompt_tokens + max_new_tokens > runtime.context_limit:
        raise ContextLimitError(
            f"Prompt tokens ({prompt_tokens}) plus requested output tokens "
            f"({max_new_tokens}) exceed the supported limit ({runtime.context_limit})."
        )

    configured_eos = runtime.model.generation_config.eos_token_id
    pad_token_id = runtime.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            configured_eos if isinstance(configured_eos, int) else configured_eos[0]
        )
    sampling = temperature > 0
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        use_cache=True,
        num_beams=1,
        num_return_sequences=1,
        repetition_penalty=1.0,
        eos_token_id=configured_eos,
        pad_token_id=pad_token_id,
        do_sample=sampling,
        temperature=temperature if sampling else 1.0,
        top_p=top_p if sampling else 1.0,
        top_k=0 if sampling else 50,
    )
    if on_prepared is not None:
        on_prepared()
    if cancellation is not None:
        cancellation.raise_if_cancelled()

    output_ids = runtime.model.generate(
        **model_inputs,
        generation_config=generation_config,
        streamer=streamer,
        stopping_criteria=(
            StoppingCriteriaList([_CancellationCriteria(cancellation)])
            if cancellation is not None else None
        ),
    )
    if cancellation is not None:
        cancellation.raise_if_cancelled()
    generated_ids = output_ids[0, prompt_tokens:].tolist()
    completion = runtime.tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    finish_reason = (
        "stop"
        if generated_ids and generated_ids[-1] in runtime.eos_token_ids
        else "length"
    )
    return GenerationResult(
        text=completion,
        prompt_tokens=prompt_tokens,
        generated_ids=generated_ids,
        finish_reason=finish_reason,
    )
