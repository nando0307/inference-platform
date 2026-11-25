import importlib.util
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import modal_hf_cache as cache


class PythonAliasTests(unittest.TestCase):
    def test_modal_entrypoint_imports_without_unbundled_runtime_module(self):
        modal = types.ModuleType("modal")
        modal.App = lambda name: types.SimpleNamespace(
            function=lambda **kwargs: lambda function: function,
            local_entrypoint=lambda: lambda function: function,
        )

        image_recipe = {}

        class Image:
            @classmethod
            def from_registry(cls, *args, **kwargs):
                image_recipe.update(kwargs)
                return cls()

            def entrypoint(self, *args, **kwargs):
                return self

            def env(self, *args, **kwargs):
                return self

            def add_local_file(self, *args, **kwargs):
                return self

        modal.Image = Image
        volume_calls = []
        modal.Volume = types.SimpleNamespace(
            from_name=lambda *args, **kwargs: volume_calls.append((args, kwargs)) or object()
        )
        cache = types.ModuleType("modal_hf_cache")
        cache.MODELS = {
            "Qwen/Qwen2.5-0.5B-Instruct": "small-revision",
            "Qwen/Qwen3-1.7B": "large-revision",
        }
        saved = {name: sys.modules.get(name) for name in ("modal", "modal_hf_cache")}
        old_path = sys.path[:]
        try:
            sys.modules.update(modal=modal, modal_hf_cache=cache)
            sys.path[:] = [entry for entry in old_path if Path(entry or ".").resolve() != Path(__file__).parents[1]]
            spec = importlib.util.spec_from_file_location(
                "modal_t4_lab_isolated", Path(__file__).parents[1] / "modal_t4_lab.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path
            for name, value in saved.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value

        self.assertEqual(image_recipe["setup_dockerfile_commands"], [module.PYTHON_ALIAS_SETUP])
        self.assertEqual(volume_calls, [(("inference-huggingface-cache",),
                                         {"version": 2, "create_if_missing": True})])

        python3 = shutil.which("python3")
        self.assertIsNotNone(python3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "python3").symlink_to(python3)
            search_path = os.pathsep.join((temporary, os.environ["PATH"]))
            subprocess.run(
                ["/bin/sh", "-c", module.PYTHON_ALIAS_SETUP.removeprefix("RUN ")],
                check=True,
                env={**os.environ, "PATH": search_path},
                capture_output=True,
                text=True,
            )
            python = shutil.which("python", path=search_path)
            self.assertEqual(Path(python).resolve(), Path(python3).resolve())
            self.assertEqual(
                subprocess.check_output([python, "-c", "import sys; print(sys.version_info[:2])"], text=True),
                subprocess.check_output([python3, "-c", "import sys; print(sys.version_info[:2])"], text=True),
            )

    def test_failed_commit_cannot_reuse_local_manifest(self):
        image = MagicMock()
        image.entrypoint.return_value = image
        image.env.return_value = image
        image.add_local_file.return_value = image
        volume = MagicMock()
        volume.commit.side_effect = [RuntimeError("commit failed"), None, None]
        app = types.SimpleNamespace(
            function=lambda **kwargs: lambda fn: fn,
            local_entrypoint=lambda: lambda fn: fn,
        )
        modal = types.SimpleNamespace(
            Volume=types.SimpleNamespace(from_name=lambda *args, **kwargs: volume),
            Image=types.SimpleNamespace(from_registry=lambda *args, **kwargs: image),
            App=lambda name: app,
        )
        files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
        downloads = []

        def download(repo, revision, cache_dir):
            downloads.append(repo)
            snapshot = cache.snapshot_path(root, repo, revision)
            snapshot.mkdir(parents=True, exist_ok=True)
            for name in files:
                (snapshot / name).write_text(name)

        fail_repo = []

        def list_files(repo, revision):
            if repo in fail_repo:
                raise RuntimeError("repo unavailable")
            return files

        hub = types.SimpleNamespace(
            HfApi=lambda: types.SimpleNamespace(list_repo_files=list_files),
            snapshot_download=download,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.dict(sys.modules, {"modal": modal, "huggingface_hub": hub}), \
                 patch.object(cache, "HF_HOME", root), \
                 patch.object(cache, "MANIFEST", root / "verified-snapshots.json"), \
                 patch.dict(os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
                lab = runpy.run_path(str(Path(__file__).parents[1] / "modal_t4_lab.py"))
                diagnostic = MagicMock(return_value="ready")
                with patch.dict(lab["lab"].__globals__, {"_diagnostics": diagnostic}):
                    with self.assertRaisesRegex(RuntimeError, "commit failed"):
                        lab["lab"]("diagnostic")
                    diagnostic.assert_not_called()
                    self.assertFalse(cache.MANIFEST.exists())
                    self.assertEqual(lab["lab"]("diagnostic"), "ready")
                    self.assertTrue(cache.verify_manifest(root))
                    os.environ.pop("HF_HUB_OFFLINE")
                    os.environ.pop("TRANSFORMERS_OFFLINE")
                    self.assertEqual(lab["lab"]("diagnostic"), "ready")
                    self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                    self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

                    small = "Qwen/Qwen2.5-0.5B-Instruct"
                    large = "Qwen/Qwen3-1.7B"
                    repaired = cache.snapshot_path(root, small, cache.MODELS[small]) / "config.json"
                    repaired.unlink()
                    fail_repo.append(large)
                    with self.assertRaisesRegex(RuntimeError, "repo unavailable"):
                        lab["lab"]("diagnostic")
                    self.assertTrue(repaired.exists())
                    self.assertFalse(cache.MANIFEST.exists())
                    self.assertEqual(diagnostic.call_count, 2)
                    fail_repo.clear()
                    self.assertEqual(lab["lab"]("diagnostic"), "ready")
                    self.assertTrue(cache.verify_manifest(root))
                self.assertEqual(diagnostic.call_count, 3)
                self.assertEqual(downloads, [small, large] * 2 + [small] + [small, large])
                self.assertEqual(volume.commit.call_count, 3)
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
                self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()


