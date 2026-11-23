"""Opt-in checks against a running two-GPU deployment.

These tests never start workers or allocate GPU capacity. Point
MULTI_GPU_INTEGRATION_URL at an approved gateway and set INFERENCE_API_TOKEN
when client authentication is enabled. They are skipped in ordinary CPU runs.
"""

import asyncio
import json
import os
import unittest

import httpx

from src.vllm_engine import HIGH_MODEL, LOW_MODEL

URL = os.environ.get("MULTI_GPU_INTEGRATION_URL", "").rstrip("/")
RUN = os.environ.get("RUN_MULTI_GPU_TESTS") == "1" and bool(URL)


@unittest.skipUnless(
    RUN,
    "set RUN_MULTI_GPU_TESTS=1 and MULTI_GPU_INTEGRATION_URL on approved two-GPU hardware",
)
class TwoGpuIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        token = os.environ.get("INFERENCE_API_TOKEN")
        headers = {"authorization": f"Bearer {token}"} if token else {}
        self.client = httpx.AsyncClient(
            base_url=URL, headers=headers, timeout=300, trust_env=False,
        )
        self.addAsyncCleanup(self.client.aclose)

    async def complete(self, model):
        return await self.client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Say one word."}],
            "max_tokens": 8,
        })

    async def test_models_are_independently_addressable(self):
        for model in (LOW_MODEL, HIGH_MODEL):
            response = await self.complete(model)
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["model"], model)
            self.assertTrue(body["choices"][0]["message"]["content"])

    async def test_public_readiness_exposes_only_status(self):
        response = await self.client.get("/ready")
        self.assertIn(response.status_code, (200, 503))
        self.assertEqual(set(response.json()), {"status"})
        self.assertIn(response.json()["status"], {"ready", "not_ready"})

    async def test_concurrent_requests_to_both_models(self):
        responses = await asyncio.gather(
            self.complete(LOW_MODEL), self.complete(HIGH_MODEL),
        )
        for model, response in zip((LOW_MODEL, HIGH_MODEL), responses):
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["model"], model)

    async def test_streaming_preserves_model_identity(self):
        async with self.client.stream("POST", "/v1/chat/completions", json={
            "model": LOW_MODEL,
            "messages": [{"role": "user", "content": "Say one word."}],
            "max_tokens": 8,
            "stream": True,
        }) as response:
            self.assertEqual(response.status_code, 200)
            frames = [
                line async for line in response.aiter_lines()
                if line.startswith("data: ")
            ]
        payloads = [
            json.loads(frame[6:]) for frame in frames if frame != "data: [DONE]"
        ]
        self.assertTrue(payloads)
        self.assertTrue(all(payload["model"] == LOW_MODEL for payload in payloads))


if __name__ == "__main__":
    unittest.main()
