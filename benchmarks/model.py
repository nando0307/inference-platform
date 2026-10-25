"""Measure natural or fixed-length CPU generation, or run a separate CPU profile."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from benchmarks.common import distribution, peak_rss_bytes, runtime_metadata, write_json
from benchmarks.timing import instrument_generation, load_cached_runtime
from src import inference


def validate_workload(manifest, runtime, selected):
    if manifest.get("schema_version") != 1 or manifest.get("model") != inference.MODEL_ID:
        raise ValueError("Expected a schema-version-1 workload for the supported Qwen model.")
    revision = runtime.benchmark_model_revision
    if manifest.get("model_revision") != revision or manifest.get("tokenizer_revision") != revision:
        raise ValueError("Workload revisions differ from the actual cached model/tokenizer revision.")
    if manifest.get("generation") != {"temperature": 0, "top_p": 1}:
        raise ValueError("The workload must specify greedy temperature=0 and top_p=1.")
    expected_baseline = {
        "device": "cpu", "dtype": "float32", "workers": 1, "concurrency": 1,
        "context_limit": runtime.context_limit, "eos_counted": True,
    }
    if manifest.get("baseline") != expected_baseline:
        raise ValueError("Workload baseline differs from the local CPU float32 baseline.")
    template_hash = hashlib.sha256(runtime.tokenizer.chat_template.encode()).hexdigest()
    if manifest.get("chat_template_sha256", template_hash) != template_hash:
        raise ValueError("Workload chat template differs from the actual cached tokenizer.")
    cases = manifest["cases"]
    names = [case["name"] for case in cases]
    if not cases or len(set(names)) != len(names):
        raise ValueError("Workload cases must have unique names.")
    if selected and not set(selected).issubset(names):
        raise ValueError("A selected case is absent from the workload.")
    for case in cases:
        cap = case["max_tokens"]
        if type(cap) is not int or not 1 <= cap <= 512:
            raise ValueError("Each output cap must be an integer in 1..512.")
        formatted = runtime.tokenizer.apply_chat_template(
            case["messages"], tokenize=False, add_generation_prompt=True
        )
        count = len(runtime.tokenizer(
            formatted, add_special_tokens=False, truncation=False
        )["input_ids"])
        if type(case["prompt_tokens"]) is not int or count != case["prompt_tokens"]:
            raise ValueError("A workload prompt count differs from actual formatted tokenization.")
        if count + cap > runtime.context_limit:
            raise ValueError("A workload case exceeds the combined local context budget.")
    return [case for case in cases if not selected or case["name"] in selected]


def run_sample(runtime, case, phase, iteration, records):
    before = len(records)
    started = perf_counter()
    result = None
    failure = None
    try:
        result = inference.generate(
            runtime, case["messages"], max_new_tokens=case["max_tokens"],
            temperature=0, top_p=1,
        )
    except Exception as error:
        failure = error
    elapsed_ms = (perf_counter() - started) * 1000
    if len(records) > before:
        row = records[-1]
    else:
        row = {
            "status": "error", "prompt_tokens": None, "completion_tokens": None,
            "max_new_tokens": case["max_tokens"], "ttft_ms": None, "tpot_ms": None,
            "inter_token_ms": [], "generation_ms": None,
            "process_peak_rss_bytes": peak_rss_bytes(),
        }
    row.update(case=case["name"], phase=phase, iteration=iteration,
               request_ms=elapsed_ms, finish_reason=None, total_tokens=None)
    if failure is not None:
        row.update(
            status="error", exception_type=type(failure).__name__,
            error_code=("context_length_exceeded" if isinstance(failure, inference.ContextLimitError)
                        else "generation_failed"),
        )
    elif result is not None:
        if (row["prompt_tokens"] != result.prompt_tokens
                or row["completion_tokens"] != len(result.generated_ids)):
            raise ValueError("Measured token counts differ from the actual generation result.")
        row.update(finish_reason=result.finish_reason,
                   total_tokens=result.prompt_tokens + len(result.generated_ids))
    return row


def summarize(rows, elapsed):
    measured = [row for row in rows if row["phase"] == "measured"]
    successful = [row for row in measured if row["status"] == "success"]
    fields = (
        "request_ms", "generation_ms", "ttft_ms", "tpot_ms", "prompt_tokens",
        "completion_tokens", "process_peak_rss_bytes",
    )
    return {
        "measured_requests": len(measured),
        "successful_requests": len(successful),
        "errors": len(measured) - len(successful),
        "warmup_errors": sum(row["status"] != "success" for row in rows if row["phase"] == "warmup"),
        **{field: distribution(row[field] for row in successful) for field in fields},
        "inter_token_ms": distribution(
            interval for row in successful for interval in row["inter_token_ms"]
        ),
        "completion_tokens_per_second": (
            sum(row["completion_tokens"] for row in successful) / elapsed if elapsed > 0 else None
        ),
        "throughput_boundary": "successful output tokens / complete measured case window",
        "quantiles": "linear interpolation; small-sample p95 is exploratory",
    }


def profile_case(runtime, case, warmups, records, trace_path):
    import torch

    samples = [run_sample(runtime, case, "warmup", i, records) for i in range(warmups)]
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        samples.append(run_sample(runtime, case, "profiled", 0, records))
    # Reserve exclusively before the exporter writes, never replace earlier evidence.
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("x", encoding="utf-8"):
        pass
    profiler.export_chrome_trace(str(trace_path))
    events = sorted(profiler.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True)
    operators = [{
        "operator": event.key,
        "calls": event.count,
        "self_cpu_time_ms": event.self_cpu_time_total / 1000,
        "cpu_time_ms": event.cpu_time_total / 1000,
        "self_cpu_memory_bytes": event.self_cpu_memory_usage,
    } for event in events[:20]]
    return samples, operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("natural", "fixed"), default="natural")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--case", action="append", dest="selected")
    parser.add_argument("--profile-case", help="Separate profiling run of this case, not headline measurements.")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be nonnegative and --repeats must be positive")
    if args.profile_case and args.selected:
        parser.error("Use --profile-case alone to select the separate profiling workload")
    output_path = Path(args.output)
    trace_path = output_path.with_suffix(".trace.json")
    if output_path.exists() or (args.profile_case and trace_path.exists()):
        parser.error("Output or trace already exists; choose a new evidence path")
    with Path(args.workload).open(encoding="utf-8") as source:
        manifest = json.load(source)
    started = perf_counter()
    runtime = load_cached_runtime()
    load_ms = (perf_counter() - started) * 1000
    cases = validate_workload(
        manifest, runtime, [args.profile_case] if args.profile_case else args.selected
    )
    payload = {
        "schema_version": 1,
        "kind": "profile" if args.profile_case else "model",
        "mode": args.mode,
        "runtime": runtime_metadata(runtime),
        "workload": manifest,
        "cache_only": True,
        "load_ms": load_ms,
        "warmups": args.warmups,
        "timing_boundary": "model.generate call; excludes tokenization and final decoding",
        "request_timing_boundary": "src.inference.generate including tokenization and final decoding",
        "instrumentation": "CPU token callbacks, no downstream text streamer; overhead included",
        "generation_settings_source": "actual per-call settings in samples",
        "fixed_length_policy": (
            "fresh per-call min_new_tokens=max_new_tokens; local budgets unchanged"
            if args.mode == "fixed" else None
        ),
    }
    records = []
    samples = []
    with instrument_generation(runtime, records.append, fixed_length=args.mode == "fixed"):
        if args.profile_case:
            samples, operators = profile_case(runtime, cases[0], args.warmups, records, trace_path)
            payload.update(
                profile_case=args.profile_case,
                profiled_requests=1,
                trace_file=str(trace_path),
                top_operators=operators,
                operator_sort="self_cpu_time_ms descending",
                profiling_note="Profiler overhead is included; do not mix these timings with headline runs.",
            )
        else:
            payload.update(repeats=args.repeats, cases=[])
            for case in cases:
                case_rows = [
                    run_sample(runtime, case, "warmup", i, records) for i in range(args.warmups)
                ]
                started = perf_counter()
                for iteration in range(args.repeats):
                    case_rows.append(run_sample(runtime, case, "measured", iteration, records))
                elapsed = perf_counter() - started
                payload["cases"].append({
                    "name": case["name"], "elapsed_seconds": elapsed,
                    "summary": summarize(case_rows, elapsed),
                })
                samples.extend(case_rows)
    payload["samples"] = samples
    payload["process_peak_rss_bytes"] = peak_rss_bytes()
    write_json(output_path, payload)
    if any(row["status"] != "success" for row in samples):
        raise SystemExit("Benchmark recorded failed requests; inspect the machine-readable output.")


if __name__ == "__main__":
    main()
