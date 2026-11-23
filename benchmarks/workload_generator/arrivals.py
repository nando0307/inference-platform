"""Pure deterministic arrival-process helpers."""

import random


def arrivals(start_s: float, end_s: float, rate_rps: float, process: str, rng: random.Random):
    if rate_rps <= 0:
        return
    offset = start_s
    while True:
        if process == "constant":
            interval = 1.0 / rate_rps
        elif process == "poisson":
            interval = rng.expovariate(rate_rps)
        else:
            fraction = min(1.0, max(0.0, (offset - start_s) / max(end_s - start_s, 1e-12)))
            current_rate = rate_rps * fraction
            if current_rate <= 0:
                offset += min(1.0, end_s - offset)
                continue
            interval = 1.0 / current_rate
        offset += interval
        if offset >= end_s:
            return
        yield round(offset, 9)
