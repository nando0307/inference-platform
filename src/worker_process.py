"""Run one configured remote vLLM worker until interrupted."""

import argparse
import asyncio
import json
import signal

from .serving_config import load_config
from .vllm_engine import VLLMEngine


async def serve(config_path, worker_id):
    config = load_config(config_path)
    worker = next((row for row in config.workers if row.id == worker_id), None)
    if worker is None:
        raise ValueError("Worker ID is not present in the selected configuration.")
    engine = VLLMEngine(worker=worker)
    await engine.start(worker.model)
    revision = next(model.revision for model in config.models.values() if model.id == worker.model)
    print(json.dumps({"worker": worker.id, "model": worker.model,
                      "revision": revision, "host": worker.host,
                      "physical_gpu_uuid": worker.gpu_uuid,
                      "process_local_cuda_device": 0, "ready": True}), flush=True)
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stopped.set)
    try:
        await stopped.wait()
    finally:
        await engine.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker", required=True)
    args = parser.parse_args()
    asyncio.run(serve(args.config, args.worker))


if __name__ == "__main__":
    main()
