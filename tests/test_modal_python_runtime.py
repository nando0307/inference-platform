import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()


