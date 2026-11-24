import tempfile
import unittest
from pathlib import Path

from modal_hf_cache import inventory, verify_snapshot


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


if __name__ == "__main__":
    unittest.main()
