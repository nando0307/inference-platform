import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from modal_python_runtime import PYTHON_ALIAS_SETUP


class PythonAliasTests(unittest.TestCase):
    def test_setup_alias_runs_the_existing_python3_interpreter(self):
        python3 = shutil.which("python3")
        self.assertIsNotNone(python3)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "python3").symlink_to(python3)
            search_path = os.pathsep.join((temporary, os.environ["PATH"]))
            env = {**os.environ, "PATH": search_path}

            subprocess.run(
                ["/bin/sh", "-c", PYTHON_ALIAS_SETUP.removeprefix("RUN ")],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )

            python = shutil.which("python", path=search_path)
            python3 = shutil.which("python3", path=search_path)
            self.assertTrue(python)
            self.assertTrue(python3)
            self.assertEqual(Path(python).resolve(), Path(python3).resolve())
            code = "import sys; print(sys.version_info[:2])"
            self.assertEqual(
                subprocess.check_output([python, "-c", code], text=True),
                subprocess.check_output([python3, "-c", code], text=True),
            )


if __name__ == "__main__":
    unittest.main()
