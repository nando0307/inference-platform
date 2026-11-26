"""Pure, simulation-only placement recommendations; never controls serving workers."""

from dataclasses import asdict, dataclass
import math

from .serving_config import ServingConfig

SCHEMA_VERSION = 1


def _nonnegative(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")


@dataclass(frozen=True)
class Profile:
    name: str
    ewma_s: float
    interval_s: float
    windows: int
    improvement: float
    cooldown_s: float
    residence_s: float
    memory_reserve: float
    horizon_s: float = 300
    transition_step_s: float = 10
    max_age_s: float = 20

    def __post_init__(self):
        for name in ("ewma_s", "interval_s", "cooldown_s", "residence_s", "horizon_s", "transition_step_s", "max_age_s", "improvement", "memory_reserve"):
            _nonnegative(getattr(self, name), name)
        if self.ewma_s == 0 or self.interval_s == 0 or self.horizon_s == 0 or self.transition_step_s == 0 or self.max_age_s == 0 or type(self.windows) is not int or self.windows < 1 or self.memory_reserve >= 1:
            raise ValueError("Invalid planner profile")


# Illustrative simulation parameters, NOT production tuning.
CONSERVATIVE = Profile("conservative", 300, 60, 3, .15, 600, 900, .20, max_age_s=120)
SHORT_EXPERIMENT = Profile("short-experiment", 30, 10, 3, .15, 60, 90, .20)


@dataclass(frozen=True)
class Snapshot:
    schema_version: int
    at_s: float
    demand_at_s: float | None
    demand_rps: dict[str, float | None]
    demand_limit_rps: dict[str, float]  # explicit per-model bound for incoming demand
    capacity_at_s: float | None
    capacity_rps: dict[str, float | None]  # per replica ID; unavailable/unknown is None
    capacity_provenance: str
    worker_ready: dict[str, bool | None]
    gpu_memory: dict[str, float | None]  # replica ID -> assumed/measured GPU memory fraction
    memory_at_s: float | None
    memory_provenance: str
    demand_provenance: str = "simulated"
    shared_gpu_factor: float = 1.0  # assumed throughput multiplier when replicas share a GPU

    def __post_init__(self):
        if self.schema_version != SCHEMA_VERSION or self.capacity_provenance not in {"assumed", "simulated", "measured"} or self.memory_provenance not in {"assumed", "simulated", "measured"} or self.demand_provenance not in {"assumed", "simulated", "measured"}:
            raise ValueError("Unsupported snapshot version or provenance")
        _nonnegative(self.at_s, "at_s")
        for stamp in (self.demand_at_s, self.capacity_at_s, self.memory_at_s):
            if stamp is not None:
                _nonnegative(stamp, "sample timestamp")
                if stamp > self.at_s:
                    raise ValueError("Future observations are forbidden")
        for values in (self.demand_rps, self.capacity_rps, self.gpu_memory):
            for value in values.values():
                if value is not None:
                    _nonnegative(value, "observation")
        if set(self.demand_limit_rps) != set(self.demand_rps):
            raise ValueError("Every demand model requires an explicit upper bound")
        for model, limit in self.demand_limit_rps.items():
            _nonnegative(limit, "demand bound")
            if self.demand_rps[model] is not None and self.demand_rps[model] > limit:
                raise ValueError("Observed demand exceeds its declared bound")
        if any(value is not True and value is not False and value is not None for value in self.worker_ready.values()):
            raise ValueError("Invalid worker readiness")
        _nonnegative(self.shared_gpu_factor, "shared_gpu_factor")
        if self.shared_gpu_factor > 1 or any(value is not None and value > 1 for value in self.gpu_memory.values()):
            raise ValueError("GPU fractions must be <= 1")


@dataclass(frozen=True)
class History:
    ewma: dict[str, float] | None = None
    last_at_s: float | None = None
    candidate: str | None = None
    consecutive: int = 0
    last_change_s: float | None = None


@dataclass(frozen=True)
class Step:
    action: str
    worker: str
    impact_s: float
    available_after: tuple[str, ...]


@dataclass(frozen=True)
class Recommendation:
    schema_version: int
    at_s: float
    current: str
    candidate: str
    action: str
    reason: str
    rejected: dict[str, str]
    scores: dict[str, float]
    demand_rps: dict[str, float] | None
    provenance: dict[str, str]
    profile: dict
    transition: tuple[Step, ...]
    transition_lost_requests: float
    history: History

    def to_dict(self):
        return asdict(self)


def _workers(config):
    return {worker.id: worker for worker in config.workers}


def _feasible(config, snapshot, profile):
    workers = _workers(config)
    if any(snapshot.worker_ready.get(w.id) is not True for w in config.workers):
        return "worker unavailable or readiness unknown"
    if any(model.id not in {w.model for w in config.workers} for model in config.models.values()):
        return "required model has no replica"
    for gpu in {gpu for host in config.hosts.values() for gpu in host.gpu_uuids}:
        total = sum(snapshot.gpu_memory[w] for w in workers if workers[w].gpu_uuid == gpu)
        if total > 1 - profile.memory_reserve + 1e-9:
            return f"GPU {gpu} exceeds memory limit with reserve"
    return None


def _transition(current, target, snapshot, profile, demand):
    old, new = _workers(current), _workers(target)
    active = set(old)
    pending = set(new) - active
    removals = active - set(new)
    steps = []
    models = {model.id for model in current.models.values()}
    gpus = {gpu for host in current.hosts.values() for gpu in host.gpu_uuids}

    def valid(members):
        if not models <= { (old | new)[id].model for id in members }:
            return False
        return all(sum(snapshot.gpu_memory[id] for id in members if (old | new)[id].gpu_uuid == gpu) <= 1 - profile.memory_reserve + 1e-9 for gpu in gpus)

    if not valid(active):
        return (), None, "current placement not resource feasible"
    # Prefer additions before removals; removal first only when an available replica survives.
    while pending or removals:
        options = [("add", id, active | {id}) for id in sorted(pending)] + [("remove", id, active - {id}) for id in sorted(removals)]
        choice = next(((action, id, members) for action, id, members in options if valid(members)), None)
        if choice is None:
            return (), None, "no availability-preserving, resource-feasible transition"
        action, id, active = choice
        (pending if action == "add" else removals).remove(id)
        steps.append(Step(action, id, profile.transition_step_s, tuple(sorted(active))))
    if sum(step.impact_s for step in steps) > profile.horizon_s:
        return (), None, "transition exceeds planning horizon"
    # Charge foregone candidate requests over each step against the same horizon
    # used for the steady-state score (RPS x seconds = requests).
    target_capacity = _capacity(target, set(new), snapshot)
    loss = sum(sum(max(0, min(demand[m], target_capacity[m]) - min(demand[m], capacity[m])) * step.impact_s
                   for m in demand)
               for step, capacity in ((step, _capacity_map(old | new, step.available_after, snapshot, models)) for step in steps))
    return tuple(steps), loss, None


def _capacity_map(workers, members, snapshot, models):
    gpu_counts = {gpu: sum(workers[id].gpu_uuid == gpu for id in members)
                  for gpu in {workers[id].gpu_uuid for id in members}}
    return {model: sum(snapshot.capacity_rps[id] * (snapshot.shared_gpu_factor if gpu_counts[workers[id].gpu_uuid] > 1 else 1)
                       for id in members if workers[id].model == model) for model in models}


def _capacity(config, members, snapshot):
    return _capacity_map(_workers(config), members, snapshot, (model.id for model in config.models.values()))


def plan(configs: dict[str, ServingConfig], current: str, snapshot: Snapshot,
         history: History = History(), profile: Profile = SHORT_EXPERIMENT) -> Recommendation:
    """One decision at a window boundary using only observations at or before that boundary."""
    if current not in configs or any(name != config.placement for name, config in configs.items()):
        raise ValueError("Current placement or candidate configuration mismatch")
    if history.last_at_s is not None and snapshot.at_s <= history.last_at_s:
        raise ValueError("Observation times must strictly increase")
    base = configs[current]
    models = {model.id: model.revision for model in base.models.values()}
    workers = {id: row for config in configs.values() for id, row in _workers(config).items()}
    rejected, scores = {}, {}
    provenance = {"demand": snapshot.demand_provenance, "capacity": snapshot.capacity_provenance, "memory": snapshot.memory_provenance}

    def result(candidate, action, reason, next_history, steps=(), loss=0):
        return Recommendation(SCHEMA_VERSION, snapshot.at_s, current, candidate, action, reason,
                              rejected, scores, next_history.ewma, provenance, asdict(profile), steps, loss, next_history)

    timestamps = (snapshot.demand_at_s, snapshot.capacity_at_s, snapshot.memory_at_s)
    if any(t is None or snapshot.at_s - t > profile.max_age_s for t in timestamps):
        return result(current, "hold", "missing or stale observation", History(last_change_s=history.last_change_s))
    if (set(snapshot.demand_rps) != set(models) or any(v is None for v in snapshot.demand_rps.values())
            or any(snapshot.capacity_rps.get(id) is None or snapshot.worker_ready.get(id) is None for id in workers)
            or any(snapshot.gpu_memory.get(id) is None for id in workers)):
        return result(current, "hold", "unknown required demand, capacity, readiness or GPU memory", History(last_change_s=history.last_change_s))
    if any(config.models != base.models or config.hosts != base.hosts for config in configs.values()):
        raise ValueError("Candidate model revisions and physical GPU topology must match")
    if _feasible(base, snapshot, profile):
        return result(current, "hold", "current placement unavailable or resource infeasible", History(last_change_s=history.last_change_s))
    contiguous = history.last_at_s is not None and abs(snapshot.at_s - history.last_at_s - profile.interval_s) < 1e-6
    alpha = 1 - math.exp(-profile.interval_s / profile.ewma_s)
    demand = {m: history.ewma[m] + alpha * (v - history.ewma[m]) if contiguous and history.ewma is not None else v
              for m, v in snapshot.demand_rps.items()}
    state = History(demand, snapshot.at_s, last_change_s=history.last_change_s)
    for name, config in sorted(configs.items()):
        reason = _feasible(config, snapshot, profile)
        if reason:
            rejected[name] = reason
            continue
        capacity = _capacity(config, set(_workers(config)), snapshot)
        scores[name] = sum(min(demand[m], capacity[m]) for m in models) * profile.horizon_s
    baseline = scores[current]
    options = []
    for name in sorted(scores):
        if name == current:
            continue
        steps, loss, reason = _transition(base, configs[name], snapshot, profile, demand)
        if reason:
            rejected[name] = reason
            continue
        net = scores[name] - loss
        if net <= baseline * (1 + profile.improvement) or net <= baseline:
            rejected[name] = "net improvement below threshold"
            continue
        options.append((net, name, steps, loss))
    if not options:
        return result(current, "hold", "no eligible improvement", state)
    net, name, steps, loss = min(options, key=lambda item: (-item[0], item[1]))
    streak = history.consecutive + 1 if contiguous and history.candidate == name else 1
    state = History(demand, snapshot.at_s, name, streak, history.last_change_s)
    if streak < profile.windows:
        return result(current, "hold", "awaiting consecutive observation windows", state)
    if history.last_change_s is not None and snapshot.at_s - history.last_change_s < max(profile.cooldown_s, profile.residence_s):
        return result(current, "hold", "cooldown or minimum residence active", state)
    return result(name, "recommend", "net improvement meets threshold and stability gates",
                  History(demand, snapshot.at_s, last_change_s=snapshot.at_s), steps, loss)
