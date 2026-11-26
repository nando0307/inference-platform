"""Pure choices over the manager's already eligible, exact-model worker snapshot."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    id: str
    inflight: int
    queued: int
    weight: int


def select(candidates: tuple[Candidate, ...], policy: str, cursor: str | None,
           credits: dict[str, int]) -> tuple[str, dict[str, int]]:
    """Return a worker ID and next credits; caller commits state only after reserving."""
    if not candidates:
        raise ValueError("No eligible workers.")
    ordered = sorted(candidates, key=lambda item: item.id)
    if policy == "least-inflight":
        return min(ordered, key=lambda item: (item.inflight, item.id)).id, credits
    if policy == "least-queue-depth":
        return min(ordered, key=lambda item: (item.queued, item.id)).id, credits
    if policy == "round-robin":
        return next((item.id for item in ordered if cursor is None or item.id > cursor), ordered[0].id), credits
    if policy == "weighted-capacity":
        # Smooth weighted round-robin: only eligible workers accrue credit.
        next_credits = {item.id: credits.get(item.id, 0) + item.weight for item in ordered}
        chosen = min(ordered, key=lambda item: (-next_credits[item.id], item.id))
        next_credits[chosen.id] -= sum(item.weight for item in ordered)
        return chosen.id, next_credits
    raise ValueError(f"Unknown routing policy: {policy}")
