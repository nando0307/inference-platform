"""Pinned, offline, single-resident vLLM process ownership for the opt-in router."""

import asyncio
import hashlib
import os
import signal
import socket
import subprocess
from collections import deque
from importlib.metadata import version
from time import perf_counter, sleep

LOW_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HIGH_MODEL = "Qwen/Qwen3-1.7B"
MODEL_SPECS = {
    LOW_MODEL: {"revision": "7ae557604adf67be50417f59c2c2f167def9a775", "thinking": False},
    HIGH_MODEL: {"revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "thinking": False},
}
ENGINE_VERSION = "0.33.0"


def stop_process_group(process, *, terminate_seconds=10, kill_seconds=5):
    """Reap the leader and wait for its entire group, including surviving descendants."""
    started = perf_counter()
    cleanup = {"stopped": False, "forced": False, "error": None}

    def group_exists():
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return False
        return True

    try:
        for sig, seconds in ((signal.SIGTERM, terminate_seconds), (signal.SIGKILL, kill_seconds)):
            if not group_exists():
                cleanup["stopped"] = True
                break
            if sig == signal.SIGKILL:
                cleanup["forced"] = True
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            deadline = perf_counter() + seconds
            while group_exists() and perf_counter() < deadline:
                sleep(min(0.05, max(0, deadline - perf_counter())))
        cleanup["stopped"] = not group_exists()
        if not cleanup["stopped"]:
            cleanup["error"] = "Process group remained after SIGTERM and SIGKILL."
    except (Exception, KeyboardInterrupt) as error:
        cleanup["error"] = f"{type(error).__name__}: {error}"
    cleanup.update(returncode=process.poll(), elapsed_ms=(perf_counter() - started) * 1000)
    return cleanup


class VLLMEngine:
    """The router owns admission; this object owns one private engine process group."""

    def __init__(self, *, port=8001, startup_timeout=360):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Engine port must be in 1..65535.")
        if not 0 < startup_timeout <= 360:
            raise ValueError("Startup timeout must be in (0, 360].")
        self.port = port
        self.startup_timeout = startup_timeout
        self.base_url = f"http://127.0.0.1:{port}"
        self.model = None
        self.process = None
        # ponytail: retain the last 256 transitions; use a log sink for long-lived history.
        self.events = deque(maxlen=256)
        self.metadata = {}
        self._tokenizers = {}
        self._tokenizer_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()

    def _event(self, event, **fields):
        row = {"event": event, "at": perf_counter(), **fields}
        self.events.append(row)
        return row

    def is_alive(self):
        return self.process is not None and self.process.poll() is None

    async def prepare(self, model_id, messages, max_tokens):
        if model_id not in MODEL_SPECS:
            raise ValueError("Unsupported model.")
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("Output limit must be in 1..512.")

        def prepare_cached():
            from transformers import AutoTokenizer, GenerationConfig
            from .inference import ContextLimitError

            if model_id not in self._tokenizers:
                revision = MODEL_SPECS[model_id]["revision"]
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, revision=revision, local_files_only=True,
                )
                config = GenerationConfig.from_pretrained(
                    model_id, revision=revision, local_files_only=True,
                )
                eos = config.eos_token_id
                eos = [eos] if type(eos) is int else list(eos or [])
                if not eos or any(type(token) is not int or token < 0 for token in eos):
                    raise ValueError("The pinned checkpoint must provide valid EOS IDs.")
                self._tokenizers[model_id] = tokenizer, sorted(eos)
            tokenizer, eos = self._tokenizers[model_id]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            ids = tokenizer(formatted, add_special_tokens=False)["input_ids"]
            if len(ids) + max_tokens > 2048:
                raise ContextLimitError("Formatted prompt plus output limit exceeds 2048 tokens.")
            return {
                "prompt_tokens": len(ids), "prompt_token_ids": ids,
                "formatted_prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
                "eos_token_ids": eos,
            }

        async with self._tokenizer_lock:
            return await asyncio.to_thread(prepare_cached)

    async def start(self, model_id):
        if model_id not in MODEL_SPECS:
            raise ValueError("Unsupported model.")
        if self.process is not None:
            if self.model == model_id and self.is_alive():
                return
            raise RuntimeError("Stop the previous process group before loading a model.")
        started = perf_counter()
        self._event("load_started", model=model_id)
        try:
            import httpx
            import torch

            if version("vllm") != ENGINE_VERSION or not torch.cuda.is_available():
                raise RuntimeError("The pinned vLLM CUDA environment is required; no CPU fallback.")
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", self.port))
            revision = MODEL_SPECS[model_id]["revision"]
            command = [
                "vllm", "serve", model_id,
                "--revision", revision, "--tokenizer-revision", revision,
                "--served-model-name", model_id,
                "--host", "127.0.0.1", "--port", str(self.port),
                "--dtype", "float16", "--kv-cache-dtype", "float16",
                "--max-model-len", "2048", "--max-num-seqs", "2",
                "--max-num-batched-tokens", "256", "--gpu-memory-utilization", "0.5",
                "--enforce-eager", "--disable-prefix-caching", "--enable-chunked-prefill",
                "--no-async-scheduling", "--generation-config", "vllm",
                "--enable-request-id-headers",
            ]
            if model_id == HIGH_MODEL:
                command += [
                    "--reasoning-parser", "qwen3",
                    "--default-chat-template-kwargs", '{"enable_thinking": false}',
                ]
            self.metadata = {
                "model": model_id, "revision": revision, "tokenizer_revision": revision,
                "engine_version": ENGINE_VERSION, "launch_command": command,
                "declared_dtype": "float16", "declared_kv_cache_dtype": "float16",
                "context_limit": 2048, "max_num_seqs": 2, "thinking": False,
                "gpu": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda,
                "package_versions": {name: version(name) for name in (
                    "torch", "transformers", "tokenizers", "vllm", "httpx",
                )},
            }
            self.process = subprocess.Popen(command, start_new_session=True, env={
                **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "VLLM_NO_USAGE_STATS": "1",
            })
            async with asyncio.timeout(max(0.001, self.startup_timeout - (perf_counter() - started))):
                async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
                    while True:
                        if not self.is_alive():
                            raise RuntimeError(f"Engine exited before readiness: {self.process.returncode}.")
                        try:
                            health = await client.get(self.base_url + "/health")
                            if health.status_code == 200:
                                models = await client.get(self.base_url + "/v1/models")
                                models.raise_for_status()
                                if [item["id"] for item in models.json()["data"]] != [model_id]:
                                    raise RuntimeError("Ready engine reports the wrong served model.")
                                break
                        except httpx.TransportError:
                            pass
                        await asyncio.sleep(0.1)
            self.model = model_id
            self.metadata["startup_ms"] = (perf_counter() - started) * 1000
            self._event("load_ready", model=model_id, pid=self.process.pid,
                        startup_ms=self.metadata["startup_ms"])
        except BaseException as error:
            self._event("load_failed", model=model_id, exception_type=type(error).__name__)
            await asyncio.shield(self.stop())
            raise

    async def stop(self):
        async with self._stop_lock:
            if self.process is None:
                self.model = None
                return
            process, model = self.process, self.model
            self._event("unload_started", model=model, pid=process.pid)
            cleanup = await asyncio.to_thread(stop_process_group, process)
            self.metadata["cleanup"] = cleanup
            self._event("unload_finished", model=model, pid=process.pid, **cleanup)
            if cleanup["stopped"]:
                self.process = None
                self.model = None
            if not cleanup["stopped"] or cleanup["forced"] or cleanup["error"]:
                raise RuntimeError("Engine process group did not stop cleanly.")
