import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from src.serving_config import load_config
from src.vllm_engine import ENGINE_VERSION, VLLMEngine

ROOT = Path(__file__).resolve().parents[1]


class WorkerLaunchTests(unittest.IsolatedAsyncioTestCase):
    async def test_worker_token_is_passed_in_the_child_environment_not_argv(self):
        worker = load_config(ROOT / "configs/serving-a.json").workers[0]
        secret = "worker-service-secret"
        spawned = {}

        class FakeProcess:
            pid = 4321
            returncode = 0

            def poll(self):
                return None

        def fake_popen(command, **kwargs):
            spawned["command"] = command
            spawned["env"] = kwargs["env"]
            return FakeProcess()

        class ReadyUpstream:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def get(self, url, headers=None):
                request = httpx.Request("GET", url)
                if url.endswith("/health"):
                    return httpx.Response(200, request=request)
                return httpx.Response(200, json={"data": [{"id": worker.model}]}, request=request)

        engine = VLLMEngine(worker=worker)
        with patch.dict(os.environ, {worker.service_token_env: secret}), \
                patch("src.vllm_engine.subprocess.Popen", fake_popen), \
                patch("src.vllm_engine.version", return_value=ENGINE_VERSION), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("httpx.AsyncClient", ReadyUpstream):
            await engine.start(worker.model)

        self.assertNotIn(secret, " ".join(spawned["command"]))
        self.assertNotIn(secret, json.dumps(engine.metadata))
        self.assertEqual(spawned["env"]["VLLM_API_KEY"], secret)
        self.assertEqual(engine.metadata["launch_command"], spawned["command"])


if __name__ == "__main__":
    unittest.main()
