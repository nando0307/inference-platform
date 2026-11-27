"""Run deterministic Python-vs-Rust config, selector and admission fixtures."""
import asyncio
import copy
from dataclasses import replace
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from src.routing import Candidate, select
from src.serving_config import validate_config
from src.vllm_engine import LOW_MODEL, HIGH_MODEL

ROOT = Path(__file__).resolve().parents[1]
RUST = ROOT / "rust-control-plane"


def config_fixtures():
    for placement in "abcd":
        source = json.loads((ROOT / f"configs/serving-{placement}.json").read_text())
        yield source
        modifications = [
            lambda x: x.update(schema_version=True), lambda x: x.update(schema_version=2),
            lambda x: x.update(unexpected=1), lambda x: x["models"]["small"].update(revision="main"),
            lambda x: x["models"]["large"].update(context_tokens=True),
            lambda x: x["hosts"]["local"]["gpu_uuids"].append(x["hosts"]["local"]["gpu_uuids"][0]),
            lambda x: x["workers"][0].update(id="wrong"),
            lambda x: x["workers"][0].update(model=[]),
            lambda x: x["workers"][0].update(cuda_device=True),
            lambda x: x["workers"][0].update(gpu_uuid="GPU-" + "x" * 36),
            lambda x: x["workers"][0].update(endpoint="http://secret@127.0.0.1:8101"),
            lambda x: x["workers"][0].update(endpoint=x["workers"][1]["endpoint"] + "/"),
            lambda x: x["workers"][0].update(memory_utilization=0.91),
            lambda x: x["workers"][0].update(capacity_weight=True),
            lambda x: x["workers"][0].update(capacity_weight=0),
            lambda x: x["workers"][0].update(max_sequences=3),
            lambda x: x["workers"][0].pop("service_token_env"),
            lambda x: x["routing"].update(policy="unknown"),
            lambda x: x["routing"].update(health_timeout_s=31),
            lambda x: x["routing"].update(queue_capacity_per_worker=3),
            lambda x: x["workers"].pop(),
        ]
        for modify in modifications:
            case = copy.deepcopy(source)
            modify(case)
            yield case
        case = copy.deepcopy(source)
        case["routing"].pop("policy")
        yield case
        case = copy.deepcopy(source)
        case["workers"][0]["capacity_weight"] = 3
        case["routing"]["policy"] = "weighted-capacity"
        yield case
        case = copy.deepcopy(source)
        case["routing"]["queue_capacity_per_worker"] = 2.0
        yield case
        case = copy.deepcopy(source)
        case["workers"][0]["endpoint"] = "http://127.0.0.1:80"
        yield case
        for endpoint in ("http://127.0.0.1:0", "http://127.0.0.1:00000000000000000000000",
                         "http://127.0.0.1:00000000000000000000080",
                         "http://127.0.0.1:00000000000000000065535",
                         "http://127.0.0.1:00000000000000000065536",
                         "http://127.0.0.1:80?", "http://127.0.0.1:80#",
                         "http://127.0.0.1:8101/./", "http://127.0.0.1:8101/%2e/",
                         "http://127.0.0.1:8101\\", "http://a／b:8101",
                         "http://exa mple.com:8101", "http://a%20b:8101",
                         "http://[fe80::1%25eth0]:8101", "http://[v1.abc]:8101",
                         " \tHTTP://example.com:8101/", "http://example.com:8101/#?"):
            case = copy.deepcopy(source)
            case["workers"][0]["endpoint"] = endpoint
            yield case
        for name in ("TOKEN²", "TOKEN\u0301", "TOKEN\u200c", "℘TOKEN"):
            case = copy.deepcopy(source)
            case["workers"][0]["service_token_env"] = name
            yield case
        if placement in "bc":
            case = copy.deepcopy(source)
            case["workers"][1]["endpoint"] = "http://127.0.0.1:00000000000000000008101"
            yield case
            for first, second in (("bücher.de", "xn--bcher-kva.de"),
                                  ("0x7f000001", "127.0.0.1"),
                                  ("EXAMPLE.COM", "example.com")):
                case = copy.deepcopy(source)
                case["workers"][0]["endpoint"] = f"http://{first}:8101"
                case["workers"][1]["endpoint"] = f"http://{second}:8101"
                yield case
        case = copy.deepcopy(source)
        case["workers"][0]["id"], case["workers"][1]["id"] = case["workers"][1]["id"], case["workers"][0]["id"]
        yield case
        case = copy.deepcopy(source)
        case["hosts"]["local"]["gpu_uuids"][0] = "GPU-" + "-" * 36
        for worker in case["workers"]:
            if worker["cuda_device"] == 0:
                worker["gpu_uuid"] = "GPU-" + "-" * 36
        yield case


def routing_fixtures():
    for policy in ("least-inflight", "least-queue-depth", "round-robin", "weighted-capacity"):
        for cursor in (None, "a", "b"):
            for counts in ((0, 0, 0, 0), (2, 0, 1, 1), (0, 1, 2, 0)):
                # Deliberately unsorted snapshot.
                candidates = [dict(id="b", inflight=counts[2], queued=counts[3], weight=1),
                              dict(id="a", inflight=counts[0], queued=counts[1], weight=3)]
                yield dict(kind="select", candidates=candidates, policy=policy, cursor=cursor, credits={})
        if policy == "weighted-capacity":
            for credits in ({"a": -2, "b": 2}, {"a": 2, "b": -2}):
                yield dict(kind="select", candidates=[dict(id="a", inflight=0, queued=0, weight=3), dict(id="b", inflight=0, queued=0, weight=1)], policy=policy, cursor=None, credits=credits)
            a = dict(id="a", inflight=0, queued=0, weight=1)
            b = dict(id="b", inflight=0, queued=0, weight=3)
            for candidates, credits in (([a, b], {}), ([a], {"a": 1, "b": -1}),
                                        ([a, b], {"a": 1}), ([a, b], {"a": 2, "b": -1}),
                                        ([dict(a, weight=18446744073709551616), b], {}),
                                        ([a, b], {"a": -18446744073709551616, "b": 18446744073709551616})):
                yield dict(kind="select", candidates=candidates, policy=policy, cursor=None, credits=credits)


async def python_trace(case):
    from src.worker_manager import MultiWorkerManager
    from src.model_router import RouterError

    class Engine:
        async def prepare(self, *args):
            return {}

    config = validate_config(json.loads((ROOT / f"configs/serving-{case['placement']}.json").read_text()))
    manager = MultiWorkerManager(replace(config, policy=case["policy"]), engine_factory=lambda row: Engine())
    for w in manager.workers.values():
        w.state = "ready"
    leases, results = {}, []
    for step in case["actions"]:
        name = step["name"]
        if step["op"] == "reserve":
            try:
                lease = manager.submit(SimpleNamespace(model=step["model"], messages=[], max_tokens=1), name)
                leases[name] = lease
                results.append(dict(name=name, worker=lease.worker.definition.id))
            except RouterError as e:
                results.append(dict(name=name, error=e.code))
            await asyncio.sleep(0)
        elif step["op"] == "release":
            await manager.release(leases.pop(name), True)
            results.append(dict(name=name, released=True))
        else:
            manager.cancel_waiting(leases.pop(name))
            results.append(dict(name=name, canceled=True))
    for lease in leases.values():
        manager.cancel_waiting(lease)
    await asyncio.gather(*manager.tasks, return_exceptions=True)
    return dict(results=results, workers={id: dict(active=len(w.active), queued=len(w.queue),
        completed=w.requests["completed"], canceled=w.requests["canceled"], rejected=w.requests["rejected"])
        for id, w in manager.workers.items()}, outcomes=manager.routing_outcomes)


def fixtures():
    for raw in config_fixtures():
        try:
            validate_config(raw)
            expected = {"valid": True}
        except (ValueError, KeyError, TypeError):
            expected = {"valid": False}
        yield dict(kind="config", raw=raw), expected
    for case in routing_fixtures():
        chosen, credits = select(tuple(Candidate(**c) for c in case["candidates"]), case["policy"], case["cursor"], case["credits"])
        yield case, dict(chosen=chosen, credits=credits)
    for policy in ("least-inflight", "least-queue-depth", "round-robin", "weighted-capacity"):
        for placement in ("b", "d"):
            actions = [dict(op="reserve", name=str(i), model=LOW_MODEL if i < 9 else HIGH_MODEL)
                       for i in range(11)]
            actions += [dict(op="release", name="0"), dict(op="cancel", name="3"),
                        dict(op="reserve", name="11", model=LOW_MODEL)]
            case = dict(kind="trace", policy=policy, placement=placement, actions=actions)
            yield case, asyncio.run(python_trace(case))


def main():
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", str(RUST / "Cargo.toml"), "--bin", "parity"], check=True)
    pairs = list(fixtures())
    proc = subprocess.run([str(RUST / "target/debug/parity")], input="".join(json.dumps(case) + "\n" for case, _ in pairs), text=True, capture_output=True, check=True)
    actual = [json.loads(line) for line in proc.stdout.splitlines()]
    assert len(actual) == len(pairs)
    for i, ((case, expected), got) in enumerate(zip(pairs, actual)):
        assert expected == got, f"fixture {i}: {case!r}\nPython {expected!r}\nRust {got!r}"
    print(f"Python/Rust parity: {len(pairs)} fixtures passed")


if __name__ == "__main__":
    main()
