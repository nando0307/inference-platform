import json
import tempfile
import unittest
from pathlib import Path

from modal_hf_cache import MODELS, inventory, verify_manifest, verify_snapshot


class SnapshotInventoryTests(unittest.TestCase):
    def test_complete_inventory_and_safe_volume_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "volume/hub/snapshot"
            blobs = root / "volume/hub/blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
                (blobs / name).write_text(name)
                (snapshot / name).symlink_to(blobs / name)
            expected = {"config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"}
            self.assertEqual(set(verify_snapshot(snapshot, root / "volume", expected)), expected)
            (snapshot / "unexpected.txt").write_text("extra")
            with self.assertRaisesRegex(ValueError, "extra"):
                verify_snapshot(snapshot, root / "volume", expected)
            (snapshot / "unexpected.txt").unlink()
            (snapshot / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                verify_snapshot(snapshot, root / "volume", expected)
            (blobs / "config.json").write_text("config.json")
            (snapshot / "config.json").symlink_to(blobs / "config.json")
            outside = root / "outside"
            outside.write_text("unsafe")
            (snapshot / "escape").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "escapes Volume"):
                inventory(snapshot, root / "volume")


    def test_verify_manifest_accepts_complete_pinned_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {}
            approved = {
                "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
                "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            }
            self.assertEqual(MODELS, approved)
            for repo, revision in approved.items():
                snapshot = root / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / revision
                snapshot.mkdir(parents=True)
                paths = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
                for name in paths:
                    (snapshot / name).write_text(name)
                manifest[repo] = {"revision": revision, "paths": paths}
            (root / "verified-snapshots.json").write_text(json.dumps(manifest))
            self.assertTrue(verify_manifest(root))
    def test_verify_manifest_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "cache invalid; re-run a T4 lab command"):
                verify_manifest(Path(temporary))

    def test_verify_manifest_corrupt_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "verified-snapshots.json").write_text("not json")
            with self.assertRaisesRegex(RuntimeError, "cache invalid; re-run a T4 lab command"):
                verify_manifest(root)

    def test_verify_manifest_missing_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "verified-snapshots.json").write_text(
                json.dumps({"some-other-repo": {"revision": "abc", "paths": []}})
            )
            with self.assertRaisesRegex(RuntimeError, "cache invalid; re-run a T4 lab command"):
                verify_manifest(root)
    def test_verify_manifest_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {}
            for repo, _rev in MODELS.items():
                manifest[repo] = {"revision": "wrong-revision", "paths": ["config.json"]}
            (root / "verified-snapshots.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "manifest revision mismatch"):
                verify_manifest(root)


if __name__ == "__main__":
    unittest.main()
