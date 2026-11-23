"""Validated JSON scenario configuration for reproducible workloads."""

from dataclasses import dataclass
import json
from pathlib import Path

from src.vllm_engine import HIGH_MODEL, LOW_MODEL

MODELS = (LOW_MODEL, HIGH_MODEL)
ARRIVAL_PROCESSES = {"constant", "poisson", "ramp"}
INPUT_PROFILES = {"short", "medium", "long", "mixed"}
OUTPUT_PROFILES = {"short", "medium", "long", "mixed"}


@dataclass(frozen=True)
class Phase:
    id: str
    start_s: float
    end_s: float
    rate_rps: float
    model_probabilities: dict[str, float]
    arrival_process: str
    input_profile: str
    output_profile: str
    stream_probability: float
    cancellation_probability: float


@dataclass(frozen=True)
class Scenario:
    schema_version: int
    id: str
    duration_s: float
    seed: int
    arrival_process: str
    phases: tuple[Phase, ...]
    prompt_corpus_id: str
    request_timeout_s: float
    max_inflight: int
    max_connections: int


def _number(value, name, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < minimum:
        raise ValueError(f"{name} must be a number >= {minimum}.")
    return float(value)


def _probability(value, name):
    value = _number(value, name)
    if value > 1:
        raise ValueError(f"{name} must be in [0, 1].")
    return value


def validate_scenario(raw: dict) -> Scenario:
    if not isinstance(raw, dict):
        raise ValueError("Scenario must be a JSON object.")
    required = {"schema_version", "id", "duration_s", "seed", "arrival_process", "phases", "prompt_corpus_id", "request_timeout_s", "max_inflight", "max_connections"}
    if set(raw) != required:
        raise ValueError(f"Scenario fields must be exactly {sorted(required)}.")
    if raw["schema_version"] != 1 or not isinstance(raw["id"], str) or not raw["id"]:
        raise ValueError("schema_version must be 1 and id must be non-empty.")
    duration = _number(raw["duration_s"], "duration_s", minimum=0.001)
    if raw["arrival_process"] not in ARRIVAL_PROCESSES:
        raise ValueError(f"Unsupported arrival_process: {raw['arrival_process']}")
    if not isinstance(raw["seed"], int) or isinstance(raw["seed"], bool):
        raise ValueError("seed must be an integer.")
    if not isinstance(raw["phases"], list) or not raw["phases"]:
        raise ValueError("phases must be a non-empty list.")
    phases = []
    previous_end = 0.0
    for row in raw["phases"]:
        if not isinstance(row, dict) or set(row) != {"id", "start_s", "end_s", "rate_rps", "model_probabilities", "arrival_process", "input_profile", "output_profile", "stream_probability", "cancellation_probability"}:
            raise ValueError("Each phase must use the complete phase schema.")
        start = _number(row["start_s"], "phase.start_s")
        end = _number(row["end_s"], "phase.end_s")
        if start != previous_end or end <= start or end > duration:
            raise ValueError("Phases must be contiguous, increasing, and within duration_s.")
        probabilities = row["model_probabilities"]
        if set(probabilities or {}) != set(MODELS):
            raise ValueError("model_probabilities must define both pinned models.")
        probabilities = {model: _probability(probabilities[model], f"{model} probability") for model in MODELS}
        if abs(sum(probabilities.values()) - 1.0) > 1e-9:
            raise ValueError("model_probabilities must sum to 1.")
        process = row["arrival_process"]
        if process not in ARRIVAL_PROCESSES:
            raise ValueError(f"Unsupported phase arrival_process: {process}")
        if row["input_profile"] not in INPUT_PROFILES or row["output_profile"] not in OUTPUT_PROFILES:
            raise ValueError("Unsupported input or output profile.")
        phases.append(Phase(row["id"], start, end, _number(row["rate_rps"], "phase.rate_rps"), probabilities, process, row["input_profile"], row["output_profile"], _probability(row["stream_probability"], "stream_probability"), _probability(row["cancellation_probability"], "cancellation_probability")))
        previous_end = end
    if previous_end != duration:
        raise ValueError("The final phase must end at duration_s.")
    for name in ("max_inflight", "max_connections"):
        if not isinstance(raw[name], int) or isinstance(raw[name], bool) or raw[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    timeout = _number(raw["request_timeout_s"], "request_timeout_s", minimum=0.001)
    return Scenario(1, raw["id"], duration, raw["seed"], raw["arrival_process"], tuple(phases), raw["prompt_corpus_id"], timeout, raw["max_inflight"], raw["max_connections"])


def load_scenario(path: str | Path) -> Scenario:
    with Path(path).open(encoding="utf-8") as handle:
        return validate_scenario(json.load(handle))
