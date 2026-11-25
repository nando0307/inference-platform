"""Bounded summaries and selected vLLM Prometheus samples; no request payloads."""

import math
import re
from collections import deque


class DurationWindow:
    def __init__(self, size=256):
        self.samples = deque(maxlen=size)

    def record(self, value):
        if value is not None:
            self.samples.append(value)

    def snapshot(self):
        ordered = sorted(self.samples)

        def percentile(fraction):
            return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1) + 0.5))] if ordered else None

        return {"window": self.samples.maxlen, "count": len(ordered),
                "last": self.samples[-1] if ordered else None,
                "p50": percentile(0.5), "p95": percentile(0.95),
                "max": ordered[-1] if ordered else None}


# Supported scheduler gauges, KV fraction, and preemption counter; alternate
# names depend on the upstream version. Duplicate series remain ambiguous.
_VLLM_NAMES = frozenset({"num_requests_running", "num_requests_waiting",
                         "kv_cache_usage_perc", "gpu_cache_usage_perc",
                         "num_preemptions", "num_preemptions_total"})
_SAMPLE = re.compile(r'^vllm:([a-z_]+)(?:\{([^{}]*)\})?\s+([\d.eE+-]+)(?:\s+\S+)?$')
_LABEL = re.compile(r'(?:^|,)\s*model_name="((?:[^"\\]|\\.)*)"(?:,|$)')


def parse_vllm_metrics(text, model):
    """Select finite, unambiguous worker-level values from an upstream scrape."""
    samples = {}
    ambiguous = set()
    for line in text.splitlines():
        if not line.startswith("vllm:"):
            continue
        match = _SAMPLE.fullmatch(line)
        if match is None or match[1] not in _VLLM_NAMES:
            continue
        labels = match[2] or ""
        label = _LABEL.search(labels)
        if label is not None and label[1] != model:
            continue
        try:
            value = float(match[3])
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0 or (match[1].endswith("cache_usage_perc") and value > 1):
            continue
        name = match[1]
        if name in samples:
            ambiguous.add(name)
        samples[name] = value
    for name in ambiguous:
        del samples[name]
    return samples
