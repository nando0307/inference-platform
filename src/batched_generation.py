"""Compare two sequential CPU completions with one static batch, without timing."""

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import GenerationConfig

from benchmarks.common import write_json
from benchmarks.warm import describe_runtime
from benchmarks.workloads import SHORT_PROMPT
from src.inference import ContextLimitError, GenerationResult, Runtime, generate, load_runtime

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
CONVERSATIONS = [
    [{"role": "user", "content": "Reply with exactly one word: Hello"}],
    [{"role": "user", "content": SHORT_PROMPT}],
]


@torch.inference_mode()
def generate_batch(
    runtime: Runtime,
    conversations: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
) -> list[GenerationResult]:
    """Greedy static generation; preserve input order and count EOS, not trailing padding."""
    if not conversations or any(not messages for messages in conversations):
        raise ValueError("Provide at least one nonempty conversation.")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 512:
        raise ValueError("max_new_tokens must be an integer between 1 and 512.")
    if runtime.tokenizer.pad_token_id is None:
        raise ValueError("Batched tokenization requires a configured padding token.")
    texts = [
        runtime.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        for messages in conversations
    ]
    # Per-call padding leaves the shared tokenizer's configuration unchanged.
    inputs = runtime.tokenizer(
        texts, padding=True, padding_side="left", return_tensors="pt",
        add_special_tokens=False, return_attention_mask=True, truncation=False,
    ).to(runtime.model.device)
    padded_width = inputs["input_ids"].shape[1]
    if padded_width + max_new_tokens > runtime.context_limit:
        raise ContextLimitError(
            f"Padded prompt width ({padded_width}) plus requested output tokens "
            f"({max_new_tokens}) exceed the supported limit ({runtime.context_limit})."
        )
    prompt_counts = inputs["attention_mask"].sum(dim=1).tolist()
    # Match the existing single-request greedy settings without changing its API.
    config = GenerationConfig(
        max_new_tokens=max_new_tokens, use_cache=True, num_beams=1,
        num_return_sequences=1, repetition_penalty=1.0,
        eos_token_id=runtime.model.generation_config.eos_token_id,
        pad_token_id=runtime.tokenizer.pad_token_id,
        do_sample=False, temperature=1.0, top_p=1.0, top_k=50,
    )
    output_ids = runtime.model.generate(**inputs, generation_config=config)
    results = []
    for prompt_count, row in zip(prompt_counts, output_ids[:, padded_width:].tolist()):
        finish_reason = "length"
        for index, token_id in enumerate(row):
            if token_id in runtime.eos_token_ids:
                # EOS can also be the pad ID: keep the first generated EOS only.
                row = row[:index + 1]
                finish_reason = "stop"
                break
        results.append(GenerationResult(
            text=runtime.tokenizer.decode(
                row, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            ),
            prompt_tokens=prompt_count,
            generated_ids=row,
            finish_reason=finish_reason,
        ))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args(argv)
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        parser.error("Set HF_HUB_OFFLINE=1; this experiment uses already-cached artifacts.")
    if not 1 <= args.max_new_tokens <= 512:
        parser.error("--max-new-tokens must be between 1 and 512.")
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a fresh output path; previous evidence is never overwritten.")
    payload = {
        "schema_version": 1,
        "kind": "cpu_static_batch_correctness",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "conversations": CONVERSATIONS,
        "max_new_tokens": args.max_new_tokens,
        "padding_side": "left",
        "performance_measured": False,
        "runtime": None,
        "sequential": {"batch_size": 1, "results": []},
        "batched": {"batch_size": len(CONVERSATIONS), "results": []},
        "comparisons": [],
        "all_token_ids_equal": None,
    }
    try:
        torch.set_num_threads(6)
        runtime = load_runtime(revision=REVISION)
        if runtime.model.device.type != "cpu" or runtime.model.dtype != torch.float32:
            raise ValueError("This local experiment requires CPU float32.")
        runtime.benchmark_model_revision = REVISION
        payload["runtime"] = describe_runtime(runtime, {"max_tokens": args.max_new_tokens})
        # Batch size belongs to each execution mode, not the shared model runtime.
        del payload["runtime"]["batch_size"]
        for messages in CONVERSATIONS:
            result = generate(
                runtime, messages, max_new_tokens=args.max_new_tokens,
                temperature=0, top_p=1,
            )
            payload["sequential"]["results"].append({
                **asdict(result), "completion_tokens": len(result.generated_ids),
            })
        results = generate_batch(runtime, CONVERSATIONS, max_new_tokens=args.max_new_tokens)
        payload["batched"]["results"] = [
            {**asdict(result), "completion_tokens": len(result.generated_ids)} for result in results
        ]
        for index, (single, batched) in enumerate(zip(payload["sequential"]["results"], results)):
            payload["comparisons"].append({
                "input_index": index,
                "prompt_tokens_equal": single["prompt_tokens"] == batched.prompt_tokens,
                "generated_ids_equal": single["generated_ids"] == batched.generated_ids,
                "finish_reason_equal": single["finish_reason"] == batched.finish_reason,
            })
        payload["all_token_ids_equal"] = all(
            row["generated_ids_equal"] for row in payload["comparisons"]
        )
        # Completion means the experiment ran, not that every comparison matched.
        payload["status"] = "completed"
    except KeyboardInterrupt:
        payload.update(status="interrupted", exception_type="KeyboardInterrupt")
    except Exception as error:
        payload["exception_type"] = type(error).__name__
    write_json(output, payload)
    print(json.dumps(payload, indent=2))
    print(f"Saved {output}")
    return 0 if payload["status"] == "completed" else 130 if payload["status"] == "interrupted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
