"""Serve the unchanged local API with opt-in, request-correlated model timing."""

import argparse
import asyncio
import json
import logging
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from threading import Lock, local
from time import perf_counter
from unittest.mock import patch

import uvicorn

from benchmarks.common import runtime_metadata
from benchmarks.timing import instrument_generation, load_cached_runtime
from src import api


class SchedulingMetrics(logging.Handler):
    """Capture only safe terminal scheduling metadata, never arbitrary log text."""

    def __init__(self, write_record):
        super().__init__()
        self.write_record = write_record

    def emit(self, record):
        try:
            event = json.loads(record.getMessage())
        except (TypeError, ValueError):
            return
        if not isinstance(event, dict) or event.get("event") != "scheduling_finished":
            return
        fields = (
            "request_id", "model", "stream", "http_status", "state", "queue_wait_ms",
            "cancellation_delay_ms", "cancellation_reason", "error_code", "exception_type",
        )
        self.write_record({
            "type": "scheduling",
            **{key: event[key] for key in fields if key in event},
        })


def serve(port, metrics_path):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    context = local()
    original_worker = api._run_inference
    original_lifespan = api.app.router.lifespan_context

    def correlated_worker(
        runtime: api.Runtime,
        job: api._InferenceJob,
        loop: asyncio.AbstractEventLoop,
    ) -> api.GenerationResult:
        context.job = job
        try:
            return original_worker(runtime, job, loop)
        finally:
            del context.job

    with metrics_path.open("x", encoding="utf-8") as metrics:
        write_lock = Lock()

        def write_record(record):
            with write_lock:
                metrics.write(json.dumps(record, allow_nan=False) + "\n")
                metrics.flush()

        def record_generation(row):
            job = context.job
            write_record({
                "type": "generation",
                "request_id": job.request_log.request_id if job.request_log is not None else None,
                **row,
                "queue_wait_ms": (
                    (job.started_at - job.admitted_at) * 1000
                    if job.started_at is not None else None
                ),
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
                handler = SchedulingMetrics(write_record)
                api.logger.addHandler(handler)
                hooks.callback(handler.close)
                hooks.callback(api.logger.removeHandler, handler)
                async with original_lifespan(app):
                    runtime = app.state.runtime
                    hooks.enter_context(instrument_generation(
                        runtime, record_generation, get_cancellation=lambda: context.job.cancellation
                    ))
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
                        "model_concurrency": 1,
                        "queue_capacity": api.QUEUE_CAPACITY,
                        "scheduling_boundary": "validated admission to worker start; cancellation signal to actual worker exit",
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
