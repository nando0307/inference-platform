"""Reproduce CPU-only baselines; writes raw latency JSONL and summary JSON to output dir."""
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKLOADS = {"selector": (1000, 10000), "reservation": (300, 3000), "synthetic": (50, 500)}
REPEATS = 5


def command(args):
    return subprocess.check_output(args, text=True).strip()


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cargo", "build", "--release", "--locked", "--manifest-path", str(ROOT / "Cargo.toml"), "--bin", "bench"], check=True)
    metadata = dict(platform=platform.platform(), cpu=command(["sysctl", "-n", "machdep.cpu.brand_string"]),
                    cpu_count=command(["sysctl", "-n", "hw.ncpu"]), memory_bytes=command(["sysctl", "-n", "hw.memsize"]),
                    python=command([sys.executable, "--version"]), rustc=command(["rustc", "--version"]),
                    cargo=command(["cargo", "--version"]), repeats=REPEATS, workloads=WORKLOADS)
    summary = {"metadata": metadata, "results": {}}
    with (output / "raw.jsonl").open("w") as raw:
        for kind, (warmup, iterations) in WORKLOADS.items():
            for language, binary in (("python", [sys.executable, str(ROOT / "bench.py")]),
                                     ("rust", [str(ROOT / "target/release/bench")])):
                cmd = ["/usr/bin/time", "-l", *binary, kind, str(warmup), str(iterations), str(REPEATS)]
                run = subprocess.run(cmd, check=True, text=True, capture_output=True)
                rss = int(re.search(r"(\d+)\s+maximum resident set size", run.stderr).group(1))
                records = [json.loads(line) for line in run.stdout.splitlines()]
                assert len(records) == REPEATS
                assert all(len(row["samples_ns"]) == iterations for row in records)
                assert all(row["rejected"] == (iterations if kind == "synthetic" else 0) for row in records)
                for row in records:
                    raw.write(json.dumps(dict(language=language, **row), separators=(",", ":")) + "\n")
                values = sorted(ns for row in records for ns in row["samples_ns"])
                elapsed = sum(row["elapsed_ns"] for row in records)
                count = iterations * REPEATS * (9 if kind == "synthetic" else 1)
                summary["results"][f"{kind}/{language}"] = {
                    "p50_ns": values[math.ceil(.50 * len(values)) - 1],
                    "p95_ns": values[math.ceil(.95 * len(values)) - 1],
                    "p99_ns": values[math.ceil(.99 * len(values)) - 1],
                    "requests_per_second": round(count * 1e9 / elapsed, 1),
                    "peak_rss_bytes": rss,
                    "max_synthetic_capacity": max(row["capacity"] for row in records),
                    "rejected": sum(row["rejected"] for row in records),
                    "total_elapsed_ns": elapsed,
                }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
