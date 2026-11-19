import asyncio
import builtins
import json
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

from langfuse import Langfuse, propagate_attributes
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src import telemetry
from src.vllm_engine import LOW_MODEL


CONFIG = {
    "LANGFUSE_TRACING_ENABLED": "true",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-telemetry-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-PRIVATE-SECRET",
    "LANGFUSE_BASE_URL": "http://127.0.0.1:1",
}
PREFIX = "langfuse.observation."


class ConfigurationTests(unittest.TestCase):
    def test_disabled_and_incomplete_configuration_never_import_sdk(self):
        original_import = builtins.__import__

        def no_sdk(name, *args, **kwargs):
            if name == "langfuse" or name.startswith("opentelemetry"):
                self.fail("Disabled tracing imported its SDK")
            return original_import(name, *args, **kwargs)

        configurations = [{}, {**CONFIG, "LANGFUSE_TRACING_ENABLED": "false"}]
        configurations.extend(
            {key: value for key, value in CONFIG.items() if key != missing}
            for missing in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
        )
        configurations.extend(
            {**CONFIG, "LANGFUSE_BASE_URL": value}
            for value in (
                "not-a-url", "file:///PRIVATE", "https://user:PRIVATE@host",
                "https://host?secret=PRIVATE", "https://host:invalid", "https://host/\nPRIVATE",
            )
        )
        with patch("builtins.__import__", side_effect=no_sdk):
            for config in configurations:
                with self.subTest(config=config), patch.dict(os.environ, config, clear=True):
                    self.assertIsNone(telemetry.create_tracer())

    def test_initialization_failure_warning_does_not_expose_credentials(self):
        with (
            patch.dict(os.environ, CONFIG, clear=True),
            patch("langfuse.Langfuse", side_effect=RuntimeError(CONFIG["LANGFUSE_SECRET_KEY"])),
            patch.object(telemetry, "_provider", SimpleNamespace()),
            patch.object(telemetry, "_configuration", None),
            self.assertLogs(telemetry.logger, level="WARNING") as captured,
        ):
            self.assertIsNone(telemetry.create_tracer())
        self.assertNotIn(CONFIG["LANGFUSE_SECRET_KEY"], " ".join(captured.output))
        self.assertNotIn("Traceback", " ".join(captured.output))


class ObservationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.exporter = InMemorySpanExporter()
        cls.global_provider = otel_trace.get_tracer_provider()
        cls.config = {**CONFIG, "LANGFUSE_PUBLIC_KEY": "pk-lf-" + uuid4().hex}
        cls.env_patch = patch.dict(os.environ, cls.config, clear=True)
        cls.provider_patch = patch.object(telemetry, "_provider", None)
        cls.configuration_patch = patch.object(telemetry, "_configuration", None)
        cls.sdk_patch = patch(
            "langfuse.Langfuse",
            side_effect=lambda **kwargs: Langfuse(**kwargs, span_exporter=cls.exporter),
        )
        for active_patch in (cls.env_patch, cls.provider_patch, cls.configuration_patch, cls.sdk_patch):
            active_patch.start()
            cls.addClassCleanup(active_patch.stop)
        cls.client = telemetry.create_tracer()
        if cls.client is None:
            raise AssertionError("Real SDK client did not initialize")
        cls.addClassCleanup(telemetry._provider.shutdown)
        cls.addClassCleanup(cls.client.shutdown)

    async def asyncSetUp(self):
        self.exporter.clear()

    async def spans(self):
        await telemetry.close_tracer(self.client)
        return self.exporter.get_finished_spans()

    async def test_real_span_correlates_request_and_excludes_ambient_private_content(self):
        request_id = uuid4().hex
        marker = "PRIVATE-PROMPT-OUTPUT-HEADERS"
        unrelated = TracerProvider()
        self.addCleanup(unrelated.shutdown)
        with unrelated.get_tracer("other-library").start_as_current_span(
            "private-parent", attributes={"gen_ai.prompt": marker},
        ) as parent:
            with propagate_attributes(user_id=marker, metadata={"private": marker}):
                trace = telemetry.start_request(self.client, request_id, backend="cpu")
                self.assertIs(otel_trace.get_current_span(), parent)
                trace.update(model=LOW_MODEL, stream=False, prompt_tokens=43)
                trace.update(completion_tokens=2, queue_wait_ms=1.25, producer_duration_ms=20.5)
                trace.finish(outcome="completed", http_status=200)
        (span,) = await self.spans()
        attributes = dict(span.attributes)
        self.assertEqual(span.context.trace_id, int(request_id, 16))
        self.assertEqual(span.name, "chat.completion")
        self.assertEqual(attributes[PREFIX + "type"], "generation")
        self.assertEqual(attributes[PREFIX + "model.name"], LOW_MODEL)
        self.assertEqual(json.loads(attributes[PREFIX + "usage_details"]), {"input": 43, "output": 2})
        self.assertEqual(json.loads(attributes[PREFIX + "metadata.request_id"]), request_id)
        self.assertEqual(json.loads(attributes[PREFIX + "metadata.queue_wait_ms"]), 1.25)
        self.assertEqual(json.loads(attributes[PREFIX + "metadata.producer_duration_ms"]), 20.5)
        self.assertNotIn(marker, json.dumps(attributes))
        self.assertFalse(any(key.endswith((".input", ".output")) for key in attributes))
        self.assertIs(otel_trace.get_tracer_provider(), self.global_provider)

    async def test_unknown_usage_is_omitted_zero_is_real_and_partial_updates_merge(self):
        rejected = telemetry.start_request(self.client, uuid4().hex, backend="cpu")
        rejected.update(prompt_tokens=True, completion_tokens=-1)
        rejected.finish(outcome="rejected", http_status=422, error_code="invalid_request")
        partial = telemetry.start_request(self.client, uuid4().hex, backend="model-router")
        partial.update(prompt_tokens=7)
        partial.finish(outcome="disconnected", error_code="client_disconnected")
        empty = telemetry.start_request(self.client, uuid4().hex, backend="cpu")
        empty.update(prompt_tokens=0)
        empty.update(completion_tokens=0)
        empty.finish(outcome="completed", http_status=200)
        rejected_span, partial_span, empty_span = await self.spans()
        self.assertNotIn(PREFIX + "usage_details", rejected_span.attributes)
        self.assertEqual(json.loads(partial_span.attributes[PREFIX + "usage_details"]), {"input": 7})
        self.assertEqual(json.loads(empty_span.attributes[PREFIX + "usage_details"]), {"input": 0, "output": 0})

    async def test_invalid_metadata_and_error_text_never_reach_exported_span(self):
        marker = "PRIVATE-EXCEPTION-CONTENT"
        trace = telemetry.start_request(self.client, uuid4().hex, backend="cpu")
        trace.update(
            model=marker, stream=marker, prompt_tokens=marker, completion_tokens=1.5,
            queue_wait_ms=float("nan"), producer_duration_ms=float("inf"),
        )
        trace.finish(outcome=marker, http_status=True, error_code=marker)
        (span,) = await self.spans()
        attributes = dict(span.attributes)
        self.assertNotIn(marker, json.dumps(attributes))
        for suffix in ("model.name", "usage_details", "metadata.stream", "metadata.http_status",
                       "metadata.error_code", "metadata.queue_wait_ms", "metadata.producer_duration_ms"):
            self.assertNotIn(PREFIX + suffix, attributes)
        self.assertEqual(json.loads(attributes[PREFIX + "metadata.outcome"]), "failed")
        self.assertEqual(span.status.status_code, otel_trace.StatusCode.ERROR)
        self.assertEqual(span.status.description, "failed")

    async def test_ending_is_once_and_late_producer_usage_cannot_change_disconnect(self):
        trace = telemetry.start_request(self.client, uuid4().hex, backend="cpu")
        trace.finish(outcome="disconnected")
        trace.update(prompt_tokens=43, completion_tokens=2, producer_duration_ms=90)
        trace.finish(outcome="completed", http_status=200)
        (span,) = await self.spans()
        self.assertEqual(json.loads(span.attributes[PREFIX + "metadata.outcome"]), "disconnected")
        self.assertNotIn(PREFIX + "usage_details", span.attributes)
        self.assertNotIn(PREFIX + "metadata.producer_duration_ms", span.attributes)
        self.assertNotIn(PREFIX + "metadata.http_status", span.attributes)

    async def test_failed_update_still_ends_real_span_without_exception_message(self):
        trace = telemetry.start_request(self.client, uuid4().hex, backend="cpu")
        with (
            patch.object(trace._observation, "update", side_effect=RuntimeError("PRIVATE-SDK-FAILURE")),
            self.assertLogs(telemetry.logger, level="WARNING") as captured,
        ):
            trace.update(prompt_tokens=3)
            trace.finish(outcome="failed", error_code="generation_failed")
        (span,) = await self.spans()
        self.assertIsNotNone(span.end_time)
        self.assertNotIn("PRIVATE-SDK-FAILURE", " ".join(captured.output))

    async def test_restarted_lifespan_continues_exporting_without_global_provider_changes(self):
        first = telemetry.start_request(self.client, uuid4().hex, backend="cpu")
        first.finish(outcome="completed", http_status=200)
        await telemetry.close_tracer(self.client)
        restarted = telemetry.create_tracer()
        request_id = uuid4().hex
        second = telemetry.start_request(restarted, request_id, backend="model-router")
        second.finish(outcome="failed", http_status=503, error_code="engine_failed")
        first_span, second_span = await self.spans()
        self.assertEqual(second_span.context.trace_id, int(request_id, 16))
        self.assertNotEqual(first_span.context.trace_id, second_span.context.trace_id)
        self.assertEqual(second_span.status.description, "engine_failed")
        self.assertIs(otel_trace.get_tracer_provider(), self.global_provider)

    async def test_configuration_changes_cannot_export_to_a_different_project(self):
        for key, value in (
            ("LANGFUSE_PUBLIC_KEY", "pk-lf-OTHER-PROJECT"),
            ("LANGFUSE_SECRET_KEY", "sk-lf-OTHER-SECRET"),
            ("LANGFUSE_BASE_URL", "https://other.example"),
        ):
            with (
                self.subTest(key=key), patch.dict(os.environ, {key: value}),
                self.assertLogs(telemetry.logger, level="WARNING") as captured,
            ):
                client = telemetry.create_tracer()
                self.assertIsNone(client)
                self.assertIsNone(telemetry.start_request(client, uuid4().hex, backend="cpu"))
            self.assertNotIn(value, " ".join(captured.output))
        self.assertEqual(await self.spans(), ())
        request_id = uuid4().hex
        trace = telemetry.start_request(telemetry.create_tracer(), request_id, backend="cpu")
        trace.finish(outcome="completed", http_status=200)
        (span,) = await self.spans()
        self.assertEqual(span.context.trace_id, int(request_id, 16))


class FailureIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_end_and_flush_failures_are_sanitized_and_fail_open(self):
        failure = RuntimeError("PRIVATE-SDK-FAILURE")
        client = SimpleNamespace(start_observation=Mock(side_effect=failure))
        observation = SimpleNamespace(update=Mock(side_effect=failure), end=Mock(side_effect=failure))
        provider = SimpleNamespace(force_flush=Mock(side_effect=failure))
        with self.assertLogs(telemetry.logger, level="WARNING") as captured:
            self.assertIsNone(telemetry.start_request(client, uuid4().hex, backend="cpu"))
            trace = telemetry.RequestTrace(observation)
            trace.finish(outcome="failed", error_code="generation_failed")
            trace.finish(outcome="completed")
            trace.update(prompt_tokens=50)
            with patch.object(telemetry, "_provider", provider):
                await telemetry.close_tracer(client)
        self.assertEqual(observation.end.call_count, 1)
        self.assertEqual(observation.update.call_count, 1)
        self.assertNotIn("PRIVATE-SDK-FAILURE", " ".join(captured.output))
        await telemetry.close_tracer(None)

    async def test_slow_flush_is_off_loop_and_does_not_hold_lifespan_open(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        worker_threads = []

        def slow_flush(*, timeout_millis):
            worker_threads.append(threading.get_ident())
            entered.set()
            release.wait(2)
            finished.set()
            return True

        with (
            patch.object(telemetry, "_provider", SimpleNamespace(force_flush=slow_flush)),
            patch.object(telemetry, "FLUSH_TIMEOUT_SECONDS", 0.02),
            self.assertLogs(telemetry.logger, level="WARNING"),
        ):
            try:
                await telemetry.close_tracer(object())
                self.assertTrue(entered.is_set())
                self.assertFalse(finished.is_set())
                self.assertNotEqual(worker_threads, [threading.get_ident()])
            finally:
                release.set()
                self.assertTrue(await asyncio.to_thread(finished.wait, 2))


if __name__ == "__main__":
    unittest.main()
