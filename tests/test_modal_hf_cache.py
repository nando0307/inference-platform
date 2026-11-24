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

    def test_verify_manifest_missing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OSError):
                verify_manifest(Path(temporary))

    def test_verify_manifest_corrupt_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "verified-snapshots.json").write_text("not json")
            with self.assertRaises(json.JSONDecodeError):
                verify_manifest(root)

    def test_verify_manifest_missing_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "verified-snapshots.json").write_text(
                json.dumps({"some-other-repo": {"revision": "abc", "paths": []}})
            )
            with self.assertRaises(KeyError):
                verify_manifest(root)

    def test_verify_manifest_revision_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {}
            for repo, _rev in MODELS.items():
                manifest[repo] = {"revision": "wrong-revision", "paths": ["config.json"]}
            (root / "verified-snapshots.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest revision mismatch"):
                verify_manifest(root)


if __name__ == "__main__":
    unittest.main()
