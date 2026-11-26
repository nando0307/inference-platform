import json
import unittest
from pathlib import Path

from src.serving_config import dry_run, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


class ServingConfigTests(unittest.TestCase):
    def read(self, placement="a"):
        return json.loads((ROOT / f"configs/serving-{placement}.json").read_text())

    def test_all_placements_match_their_physical_worker_assignments(self):
        expected = {
            "A": {"small-0": ("Qwen/Qwen2.5-0.5B-Instruct", 0), "large-1": ("Qwen/Qwen3-1.7B", 1)},
            "B": {"small-0": ("Qwen/Qwen2.5-0.5B-Instruct", 0), "large-0": ("Qwen/Qwen3-1.7B", 0), "small-1": ("Qwen/Qwen2.5-0.5B-Instruct", 1)},
            "C": {"small-0": ("Qwen/Qwen2.5-0.5B-Instruct", 0), "large-0": ("Qwen/Qwen3-1.7B", 0), "large-1": ("Qwen/Qwen3-1.7B", 1)},
            "D": {"small-0": ("Qwen/Qwen2.5-0.5B-Instruct", 0), "large-0": ("Qwen/Qwen3-1.7B", 0)},
        }
        for placement, workers in expected.items():
            with self.subTest(placement=placement):
                config = validate_config(self.read(placement.lower()))
                mapping = {row["worker"]: (row["model"], row["cuda_device"]) for row in dry_run(config)}
                self.assertEqual(mapping, workers)
                self.assertEqual(config.models["small"].context_tokens, 2048)
                self.assertEqual(config.models["large"].context_tokens, 2048)

    def test_invalid_identity_mapping_revision_and_budget_fail_before_worker_creation(self):
        mutations = [
            lambda raw: raw.update(schema_version=2),
            lambda raw: raw.update(schema_version=True),
            lambda raw: raw["workers"][0].update(id="renamed"),
            lambda raw: raw["workers"][0].update(model=[]),
            lambda raw: raw["workers"][0].update(endpoint="http://secret@127.0.0.1:8101"),
            lambda raw: raw["workers"][0].update(gpu_uuid="GPU-00000000-0000-0000-0000-000000000002"),
            lambda raw: raw["workers"][0].update(cuda_device=1),
            lambda raw: raw["models"]["small"].update(revision="main"),
            lambda raw: raw["workers"][0].update(memory_utilization=1.1),
            lambda raw: raw["workers"][0].update(endpoint=raw["workers"][1]["endpoint"]),
            lambda raw: raw["workers"][1].update(endpoint=raw["workers"][0]["endpoint"] + "/"),
            lambda raw: raw["workers"][1].update(
                endpoint=raw["workers"][0]["endpoint"].replace("127.0.0.1", "localhost")),
            lambda raw: raw["workers"].pop(),
        ]
        for mutate in mutations:
            raw = self.read()
            mutate(raw)
            with self.subTest(raw=raw), self.assertRaises((ValueError, KeyError)):
                validate_config(raw)

    def test_per_process_utilization_is_not_accepted_as_physical_partition_above_cap(self):
        raw = self.read("b")
        raw["workers"][0]["memory_utilization"] = 0.6
        raw["workers"][1]["memory_utilization"] = 0.5
        with self.assertRaisesRegex(ValueError, "per-GPU policy cap"):
            validate_config(raw)

    def test_worker_fields_and_queue_policy_are_strict(self):
        raw = self.read()
        del raw["workers"][0]["service_token_env"]
        with self.assertRaises((KeyError, ValueError)):
            validate_config(raw)
        raw = self.read()
        raw["routing"]["queue_capacity_per_worker"] = 3
        with self.assertRaisesRegex(ValueError, "queue_capacity_per_worker"):
            validate_config(raw)

    def test_policy_defaults_weights_and_invalid_values(self):
        raw = self.read("b")
        raw["routing"].pop("policy")
        self.assertEqual(validate_config(raw).policy, "least-inflight")
        self.assertTrue(all(w.capacity_weight == 1 for w in validate_config(raw).workers))
        raw["workers"][0]["capacity_weight"] = 3
        for policy in ("least-inflight", "least-queue-depth", "round-robin", "weighted-capacity"):
            raw["routing"]["policy"] = policy
            self.assertEqual(validate_config(raw).workers[0].capacity_weight, 3)
        for value in (0, -1, 1.5, True, "3", None):
            raw["workers"][0]["capacity_weight"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "capacity_weight"):
                validate_config(raw)
        raw["workers"][0]["capacity_weight"] = 1
        raw["routing"]["policy"] = "random"
        with self.assertRaisesRegex(ValueError, "policy"):
            validate_config(raw)

    def test_loader_reads_typed_config(self):
        config = load_config(ROOT / "configs/serving-d.json")
        self.assertEqual(config.placement, "D")
        self.assertEqual(len(config.workers), 2)
        self.assertEqual({worker.cuda_device for worker in config.workers}, {0})


if __name__ == "__main__":
    unittest.main()
