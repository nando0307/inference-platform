"""Generate and persist deterministic request traces."""

import hashlib
import json
import random
from pathlib import Path

from .arrivals import arrivals
from .config import Scenario
from .prompts import PromptProvider
from src.vllm_engine import HIGH_MODEL, LOW_MODEL, MODEL_SPECS

OUTPUTS = {"short": (8, 32), "medium": (32, 128), "long": (96, 256)}


def _rng(seed: int, name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _choice(rng, probabilities):
    value = rng.random()
    for name, probability in probabilities.items():
        value -= probability
        if value < 0:
            return name
    return next(reversed(probabilities))


def generate_trace(scenario: Scenario, *, prompt_provider=None):
    provider = prompt_provider or PromptProvider(corpus_id=scenario.prompt_corpus_id)
    records = []
    sequence = 0
    for phase in scenario.phases:
        phase_arrivals = arrivals(phase.start_s, phase.end_s, phase.rate_rps, phase.arrival_process, _rng(scenario.seed, f"arrival:{phase.id}"))
        model_rng = _rng(scenario.seed, f"model:{phase.id}")
        prompt_rng = _rng(scenario.seed, f"prompt:{phase.id}")
        output_rng = _rng(scenario.seed, f"output:{phase.id}")
        stream_rng = _rng(scenario.seed, f"stream:{phase.id}")
        cancel_rng = _rng(scenario.seed, f"cancel:{phase.id}")
        for offset in phase_arrivals:
            model = _choice(model_rng, phase.model_probabilities)
            input_profile = phase.input_profile
            if input_profile == "mixed":
                input_profile = ("short", "medium", "long")[prompt_rng.randrange(3)]
            output_profile = phase.output_profile
            if output_profile == "mixed":
                output_profile = ("short", "medium", "long")[output_rng.randrange(3)]
            messages, actual_prompt_tokens = provider.prompt(model, input_profile, sequence)
            low, high = OUTPUTS[output_profile]
            max_tokens = output_rng.randint(low, high)
            if actual_prompt_tokens is not None and actual_prompt_tokens + max_tokens > 2048:
                raise ValueError(f"Prompt and output exceed the 2048-token context for {model}.")
            cancel_after = None
            if cancel_rng.random() < phase.cancellation_probability:
                cancel_after = round(cancel_rng.uniform(0.1, scenario.request_timeout_s), 3)
            records.append({
                "trace_schema_version": 1,
                "trace_id": f"{scenario.id}-seed-{scenario.seed}",
                "request_id": f"{scenario.id}-{sequence:08d}",
                "scenario_id": scenario.id,
                "phase_id": phase.id,
                "scheduled_offset_s": offset,
                "model": model,
                "model_revision": MODEL_SPECS[model]["revision"],
                "messages": messages,
                "input_profile": input_profile,
                "target_prompt_tokens": None,
                "actual_prompt_tokens": actual_prompt_tokens,
                "output_profile": output_profile,
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "stream": stream_rng.random() < phase.stream_probability,
                "request_timeout_s": scenario.request_timeout_s,
                "cancel_after_s": cancel_after,
            })
            sequence += 1
    records.sort(key=lambda row: row["scheduled_offset_s"])
    return records


def metadata(scenario: Scenario, records):
    return {
        "trace_schema_version": 1,
        "trace_id": f"{scenario.id}-seed-{scenario.seed}",
        "scenario_id": scenario.id,
        "seed": scenario.seed,
        "duration_s": scenario.duration_s,
        "request_count": len(records),
        "models": {model: {"id": model, "revision": MODEL_SPECS[model]["revision"]} for model in (LOW_MODEL, HIGH_MODEL)},
        "prompt_corpus_id": scenario.prompt_corpus_id,
        "generator_policy": "independent-sha256-derived-random-streams",
    }


def write_trace(records, meta, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_trace(path: str | Path):
    path = Path(path)
    metadata_path = path.with_suffix(".meta.json")
    if not metadata_path.exists():
        raise ValueError("Trace metadata sidecar is required.")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta.get("trace_schema_version") != 1:
        raise ValueError("Unsupported trace schema version.")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if meta.get("request_count") != len(records):
        raise ValueError("Trace metadata request_count does not match the trace.")
    if any(record.get("trace_schema_version") != 1 for record in records):
        raise ValueError("Trace contains an unsupported request schema version.")
    return records
