"""Safe deterministic prompt corpus and optional pinned-tokenizer support."""

import hashlib

from src.vllm_engine import HIGH_MODEL, LOW_MODEL, MODEL_SPECS

CORPUS_ID = "synthetic-v1"
PROFILES = {
    "short": (32, 96),
    "medium": (128, 256),
    "long": (512, 1024),
}
_BASE = (
    "Explain how an inference server processes a request, including tokenization, "
    "prefill, decoding, KV cache use, and stopping."
)


class PromptProvider:
    def __init__(self, *, corpus_id=CORPUS_ID, tokenizers=None):
        if corpus_id != CORPUS_ID:
            raise ValueError(f"Unsupported prompt corpus: {corpus_id}")
        self.corpus_id = corpus_id
        self.tokenizers = tokenizers or {}

    def prompt(self, model: str, profile: str, index: int):
        if model not in MODEL_SPECS:
            raise ValueError("Unsupported model.")
        if profile == "mixed":
            profile = ("short", "medium", "long")[index % 3]
        if profile not in PROFILES:
            raise ValueError(f"Unsupported input profile: {profile}")
        repetitions = {"short": 1, "medium": 5, "long": 18}[profile]
        text = f"Request {index}: " + ("Background context. " * repetitions) + _BASE
        messages = [{"role": "user", "content": text}]
        tokenizer = self.tokenizers.get(model)
        actual = None
        if tokenizer is not None:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            actual = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
        return messages, actual

    def identifier(self):
        return {"id": self.corpus_id, "sha256": hashlib.sha256(_BASE.encode()).hexdigest()}


def load_tokenizers(models=(LOW_MODEL, HIGH_MODEL)):
    from transformers import AutoTokenizer
    return {model: AutoTokenizer.from_pretrained(model, revision=MODEL_SPECS[model]["revision"], local_files_only=True) for model in models}
