"""Build a reproducible synthetic workload using the cached model tokenizer."""

import argparse
import hashlib
from pathlib import Path

from benchmarks.common import write_json

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SHORT_PROMPT = (
    "Explain how an inference server processes a request. Use a numbered list "
    "covering tokenization, prefill, decoding, the KV cache, and stopping."
)
BACKGROUND = (
    "An inference server shares model weights across requests, but each request "
    "has its own input tokens and KV cache. Prefill processes the input sequence. "
    "Decoding generates new tokens using the cached keys and values. A request "
    "finishes when an end-of-sequence token appears or its output limit is reached."
)
# Synthetic repeated context intentionally varies input length, not subject matter.
LONG_PROMPT = "Background notes:\n" + "\n".join([BACKGROUND] * 16) + "\n\n" + SHORT_PROMPT


def build_workload():
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    # Download/setup is a separate operation, never hidden in a benchmark run.
    snapshot = snapshot_download(MODEL_ID, local_files_only=True)
    revision = Path(snapshot).name
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True
    )
    cases = []
    context_limit = min(2048, config.max_position_embeddings)
    for input_name, prompt in (("short", SHORT_PROMPT), ("long", LONG_PROMPT)):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        count = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        for output_name, cap in (("short", 32), ("long", 128)):
            if count + cap > context_limit:
                raise ValueError("Fixed workload exceeds the supported context budget.")
            cases.append({
                "name": f"{input_name}-input-{output_name}-output",
                "messages": messages,
                "prompt_tokens": count,
                "max_tokens": cap,
            })
    return {
        "schema_version": 1,
        "model": MODEL_ID,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "description": "Fixed synthetic prompts; repeated background isolates input length, not quality.",
        "baseline": {
            "device": "cpu", "dtype": "float32", "workers": 1,
            "concurrency": 1, "context_limit": context_limit, "eos_counted": True,
        },
        "generation": {"temperature": 0, "top_p": 1},
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = build_workload()
    write_json(args.output, manifest)
    print(f"Model revision: {manifest['model_revision']}")
    for case in manifest["cases"]:
        print(f"{case['name']}: {case['prompt_tokens']} formatted input tokens, "
              f"{case['max_tokens']} maximum output tokens")


if __name__ == "__main__":
    main()
