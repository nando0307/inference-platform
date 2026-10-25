"""Small measurement helpers; importing these never imports or loads a model."""

import json
import math
import os
import platform
import resource
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def environment():
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "fastapi", "uvicorn", "httpx"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor() or None
    if sys.platform == "darwin":
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "packages": packages,
    }


def peak_rss_bytes():
    """Process lifetime RSS high-water mark, not current or accelerator memory."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Distribution values must be finite.")
    if not values:
        return dict(n=0, p50=None, p95=None, min=None, max=None)

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return dict(n=len(values), p50=percentile(.5), p95=percentile(.95),
                min=values[0], max=values[-1])


def runtime_metadata(runtime):
    import torch

    return {
        "environment": environment(),
        "model": runtime.model.config._name_or_path,
        "model_revision": runtime.benchmark_model_revision,
        "device": str(runtime.model.device),
        "dtype": str(runtime.model.dtype),
        "context_limit": runtime.context_limit,
        "eos_counted": True,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "memory_metric": "process_peak_rss_bytes",
        "memory_scope": "process lifetime high-water RSS; not current RSS or device allocation",
    }


def write_json(path, data):
    """Keep prior experiment evidence rather than overwriting it implicitly."""
    path = Path(path)
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded)
