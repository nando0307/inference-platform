"""Serve the unchanged local API with opt-in, request-correlated model timing."""

import argparse
import asyncio
import json
from _thread import LockType
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from threading import local
from time import perf_counter
from unittest.mock import patch

import uvicorn
from transformers import TextIteratorStreamer

from benchmarks.common import runtime_metadata
from benchmarks.timing import instrument_generation, load_cached_runtime
from src import api


def serve(port, metrics_path):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    context = local()
    original_worker = api._run_inference
    original_lifespan = api.app.router.lifespan_context

    def correlated_worker(
        runtime: api.Runtime,
        request: api.ChatCompletionRequest,
        guard: LockType,
        loop: asyncio.AbstractEventLoop,
        streamer: TextIteratorStreamer | None,
        prepared: asyncio.Future[None] | None,
        request_log: api._RequestLog | None,
    ) -> api.GenerationResult:
        context.request_id = request_log.request_id if request_log is not None else None
        try:
            return original_worker(runtime, request, guard, loop, streamer, prepared, request_log)
        finally:
            del context.request_id

    with metrics_path.open("x", encoding="utf-8") as metrics:
        def write_record(record):
            metrics.write(json.dumps(record, allow_nan=False) + "\n")
            metrics.flush()

        def record_generation(row):
            write_record({
                "type": "generation",
                "request_id": context.request_id,
                **row,
            })

        @asynccontextmanager
        async def benchmark_lifespan(app):
            startup_started = perf_counter()
            load_ms = None

            def cached_loader():
                nonlocal load_ms
                started = perf_counter()
                runtime = load_cached_runtime()
                load_ms = (perf_counter() - started) * 1000
                return runtime

            # Keep hooks alive through the original lifespan's worker shutdown.
            with ExitStack() as hooks:
                hooks.enter_context(patch.object(api, "load_runtime", cached_loader))
                hooks.enter_context(patch.object(api, "_run_inference", correlated_worker))
                async with original_lifespan(app):
                    runtime = app.state.runtime
                    hooks.enter_context(instrument_generation(runtime, record_generation))
                    metadata = runtime_metadata(runtime)
                    write_record({
                        "type": "startup",
                        "schema_version": 1,
                        "runtime": metadata,
                        "load_ms": load_ms,
                        "startup_ms": (perf_counter() - startup_started) * 1000,
                        "startup_boundary": "lifespan entry to instrumentation ready; excludes imports",
                        "cache_only": True,
                        "workers": 1,
                        "concurrency": 1,
                        "generation_settings_source": "actual per-call settings in generation records",
                        "timing_boundary": "model.generate call; excludes tokenization and final decoding",
                        "instrumentation": "CPU token callbacks; includes downstream streamer work",
                    })
                    yield

        with patch.object(api.app.router, "lifespan_context", benchmark_lifespan):
            uvicorn.run(api.app, host="127.0.0.1", port=port, workers=1, reload=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--metrics", required=True, help="New JSONL file; existing evidence is never overwritten.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    serve(args.port, args.metrics)


if __name__ == "__main__":
    main()
