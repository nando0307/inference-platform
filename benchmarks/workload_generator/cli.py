"""CLI for validating, generating, inspecting, and replaying workload traces."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .config import load_scenario
from .prompts import PromptProvider, load_tokenizers
from .executor import replay_trace
from .recorder import write_results, summarize_results
from .trace import generate_trace, metadata, read_trace, write_trace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--scenario", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--scenario", required=True)
    generate.add_argument("--output", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--trace", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--endpoint", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--max-inflight", type=int, default=None)
    replay.add_argument("--max-connections", type=int, default=None)
    replay.add_argument("--time-scale", type=float, default=1.0)
    replay.add_argument("--drain-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)
    if args.command == "validate":
        scenario = load_scenario(args.scenario)
        print(json.dumps({"scenario": scenario.id, "duration_s": scenario.duration_s, "phases": len(scenario.phases)}))
    elif args.command == "generate":
        scenario = load_scenario(args.scenario)
        records = generate_trace(scenario, prompt_provider=PromptProvider(corpus_id=scenario.prompt_corpus_id, tokenizers=load_tokenizers()))
        write_trace(records, metadata(scenario, records), args.output)
        print(json.dumps({"trace": args.output, "requests": len(records), "metadata": str(Path(args.output).with_suffix('.meta.json'))}))
    elif args.command == "inspect":
        records = read_trace(args.trace)
        print(json.dumps({"trace": args.trace, "requests": len(records), "first_offset_s": records[0]["scheduled_offset_s"] if records else None, "last_offset_s": records[-1]["scheduled_offset_s"] if records else None}))
    else:
        records = read_trace(args.trace)
        results = asyncio.run(replay_trace(records, args.endpoint, auth_token=os.environ.get("INFERENCE_API_TOKEN"), max_inflight=args.max_inflight or 64, max_connections=args.max_connections or 64, time_scale=args.time_scale, drain_timeout_s=args.drain_timeout))
        write_results(results, args.output, intended_count=len(records))
        print(json.dumps({"results": args.output, "summary": summarize_results(results, intended_count=len(records))}))
    return 0
