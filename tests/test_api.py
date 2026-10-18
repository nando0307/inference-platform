import asyncio
import gc
import json
import unittest

from threading import Event, Timer, get_ident
from unittest.mock import patch
from uuid import UUID

import httpx
from pydantic import ValidationError

from src.api import (
    ChatCompletionRequest,
    InferenceBusy,
    InferenceUnavailable,
    app,
    submit_inference,
)
from src.inference import ContextLimitError, GenerationResult, MODEL_ID


def request_data(**changes):
    return {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Hello"}],
        **changes,
    }


class RequestValidationTests(unittest.TestCase):
    def assert_rejected(self, data, code=None):
        # Exercise both Python callers and the JSON boundary used by HTTP clients.
        for source in (data, json.dumps(data)):
            with self.subTest(source_type=type(source).__name__):
                with self.assertRaises(ValidationError) as caught:
                    if isinstance(source, str):
                        ChatCompletionRequest.model_validate_json(source)
                    else:
                        ChatCompletionRequest.model_validate(source)
                if code is not None:
                    codes = {
                        error["type"]
                        for error in caught.exception.errors(include_input=False)
                    }
                    self.assertIn(code, codes)

    def test_valid_history_preserves_text_and_allows_consecutive_roles(self):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "  Hello\n世界!  "},
            {"role": "user", "content": "Another question."},
            {"role": "assistant", "content": "An earlier answer."},
        ]
        result = ChatCompletionRequest.model_validate_json(
            json.dumps(request_data(messages=messages))
        )
        self.assertEqual(
            [message.model_dump() for message in result.messages], messages
        )

    def test_valid_numeric_boundaries_and_integer_sampling_values(self):
        for settings in (
            {"max_tokens": 1, "temperature": 0, "top_p": 1},
            {"max_tokens": 512, "temperature": 2, "top_p": 1},
            {"temperature": 1, "top_p": 0.01},
            {"temperature": 0.7, "top_p": 0.9},
        ):
            with self.subTest(settings=settings):
                result = ChatCompletionRequest.model_validate_json(
                    json.dumps(request_data(**settings))
                )
                for field, value in settings.items():
                    self.assertEqual(getattr(result, field), value)

    def test_required_fields_and_model_identity(self):
        for field in ("model", "messages"):
            data = request_data()
            del data[field]
            self.assert_rejected(data, "missing")
        for model in ("another-model", "", 123, None):
            self.assert_rejected(request_data(model=model))

    def test_invalid_messages_and_content(self):
        for messages in (
            [],
            "Hello",
            None,
            [{"role": "tool", "content": "Hello"}],
            [{"role": "user"}],
            [{"content": "Hello"}],
            [{"role": "user", "content": ""}],
            [{"role": "user", "content": " \t\n\u2003"}],
            [{"role": "user", "content": 123}],
            [{"role": "user", "content": [{"type": "image_url"}]}],
        ):
            with self.subTest(messages=messages):
                self.assert_rejected(request_data(messages=messages))

    def test_unknown_request_and_message_fields(self):
        for field in (
            "tools",
            "n",
            "top_k",
            "stop",
            "seed",
            "stream_options",
            "unknown",
        ):
            with self.subTest(field=field):
                self.assert_rejected(request_data(**{field: None}), "extra_forbidden")
        self.assert_rejected(
            request_data(messages=[{"role": "user", "content": "Hello", "name": "x"}]),
            "extra_forbidden",
        )

    def test_numeric_types_do_not_coerce_strings_or_booleans(self):
        for field in ("max_tokens", "temperature", "top_p"):
            for value in ("1", True, False, None):
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))
        self.assert_rejected(request_data(max_tokens=1.0), "int_type")

    def test_numeric_ranges(self):
        for field, values in (
            ("max_tokens", (0, -1, 513)),
            ("temperature", (-0.1, 2.01)),
            ("top_p", (0, -0.1, 1.01)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))

    def test_nonfinite_numbers(self):
        for field in ("max_tokens", "temperature", "top_p"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    self.assert_rejected(request_data(**{field: value}))

    def test_stream_requires_boolean_and_rejects_unsupported_streaming(self):
        for value in ("false", "true", 0, 1, None):
            with self.subTest(value=value):
                self.assert_rejected(request_data(stream=value), "bool_type")
        self.assert_rejected(request_data(stream=True), "unsupported_parameter")

    def test_greedy_top_p_combination_including_default_temperature(self):
        self.assert_rejected(request_data(top_p=0.9), "invalid_parameter_combination")
        self.assert_rejected(
            request_data(temperature=0, top_p=0.9), "invalid_parameter_combination"
        )

    def test_malformed_json_and_nonobject_body(self):
        for body in ('{"model":', "[]", "null", '"text"'):
            with self.subTest(body=body):
                with self.assertRaises(ValidationError):
                    ChatCompletionRequest.model_validate_json(body)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runtime = object()
        loader_patch = patch("src.api.load_runtime", return_value=self.runtime)
        self.loader = loader_patch.start()
        self.addCleanup(loader_patch.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )
        self.addAsyncCleanup(self.client.aclose)

    async def test_startup_reuses_runtime_and_shutdown_releases_resources(self):
        async with app.router.lifespan_context(app):
            executor = app.state.executor
            for _ in range(2):
                health = await self.client.get("/health")
                ready = await self.client.get("/ready")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json(), {"status": "alive"})
                self.assertEqual(ready.status_code, 200)
                self.assertEqual(ready.json(), {"ready": True, "status": "ready"})
                self.assertIs(app.state.runtime, self.runtime)
            self.loader.assert_called_once_with()

        self.assertFalse(app.state.initialized)
        self.assertTrue(app.state.shutting_down)
        self.assertIsNone(app.state.runtime)
        self.assertIsNone(app.state.executor)
        with self.assertRaises(RuntimeError):
            executor.submit(lambda: None)

    async def test_loading_failure_prevents_serving(self):
        self.loader.side_effect = RuntimeError("Controlled loading failure")
        with self.assertRaises(RuntimeError):
            async with app.router.lifespan_context(app):
                self.fail("Startup failure must not enter the serving phase.")
        self.assertFalse(app.state.initialized)
        self.assertIsNone(app.state.runtime)
        self.assertIsNone(app.state.executor)
        ready = await self.client.get("/ready")
        self.assertEqual(ready.status_code, 503)
        self.assertFalse(ready.json()["ready"])

    async def test_executor_creation_failure_releases_loaded_runtime(self):
        with patch("src.api.ThreadPoolExecutor", side_effect=RuntimeError("No worker")):
            with self.assertRaises(RuntimeError):
                async with app.router.lifespan_context(app):
                    self.fail("Partial startup must not enter the serving phase.")
        self.assertFalse(app.state.initialized)
        self.assertIsNone(app.state.runtime)
        self.assertIsNone(app.state.executor)

    async def test_health_remains_available_when_inference_slot_is_busy(self):
        async with app.router.lifespan_context(app):
            app.state.admission_guard.acquire()
            try:
                health = await self.client.get("/health")
                ready = await self.client.get("/ready")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(ready.status_code, 503)
                self.assertEqual(ready.json(), {"ready": False, "status": "busy"})
            finally:
                app.state.admission_guard.release()
            ready = await self.client.get("/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertTrue(ready.json()["ready"])

    async def test_uninitialized_application_does_not_claim_readiness(self):
        with (
            patch.object(app.state, "initialized", False, create=True),
            patch.object(app.state, "shutting_down", False, create=True),
        ):
            ready = await self.client.get("/ready")
        self.assertEqual(ready.status_code, 503)
        self.assertEqual(ready.json(), {"ready": False, "status": "not_ready"})

    async def test_shutdown_waits_for_worker_without_blocking_http_loop(self):
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        started = Event()
        release = Event()
        shutdown_started = Event()
        executor = app.state.executor
        guard = app.state.admission_guard

        def held_producer(*args, **kwargs):
            started.set()
            release.wait()
            return app.state.runtime

        generation_patch = patch("src.api.generate", side_effect=held_producer)
        generation_patch.start()
        self.addCleanup(generation_patch.stop)
        future = submit_inference(app, ChatCompletionRequest.model_validate(request_data()))
        original_shutdown = executor.shutdown

        def observed_shutdown(*args, **kwargs):
            shutdown_started.set()
            return original_shutdown(*args, **kwargs)

        # Release even if a regression blocks the event loop during shutdown.
        safety_release = Timer(5, release.set)
        safety_release.start()
        shutdown_task = None
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            with patch.object(executor, "shutdown", side_effect=observed_shutdown):
                shutdown_task = asyncio.create_task(context.__aexit__(None, None, None))
                self.assertTrue(await asyncio.to_thread(shutdown_started.wait, 2))
                self.assertFalse(shutdown_task.done())
                self.assertFalse(future.done())
                self.assertIs(app.state.runtime, self.runtime)
                self.assertFalse(app.state.initialized)
                health = await self.client.get("/health")
                ready = await self.client.get("/ready")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(ready.status_code, 503)
                self.assertEqual(
                    ready.json(), {"ready": False, "status": "shutting_down"}
                )
                with self.assertRaises(InferenceUnavailable):
                    submit_inference(
                        app, ChatCompletionRequest.model_validate(request_data())
                    )
                release.set()
                await asyncio.wait_for(shutdown_task, timeout=3)
            self.assertIs(future.result(), self.runtime)
            self.assertIsNone(app.state.runtime)
            self.assertIsNone(app.state.executor)
            self.assertEqual(app.state.producer_futures, set())
            self.assertFalse(guard.locked())
        finally:
            release.set()
            safety_release.cancel()
            safety_release.join()
            if shutdown_task is None:
                await context.__aexit__(None, None, None)
            else:
                await shutdown_task


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        loader_patch = patch("src.api.load_runtime", return_value=object())
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        self.addAsyncCleanup(context.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.request = ChatCompletionRequest.model_validate(request_data())
        self.result = GenerationResult("answer", 3, [7], "length")

    async def test_unavailable_state_rejects_before_submitting_work(self):
        for initialized, shutting_down in ((False, False), (True, True)):
            with self.subTest(initialized=initialized, shutting_down=shutting_down):
                with (
                    patch.object(app.state, "initialized", initialized),
                    patch.object(app.state, "shutting_down", shutting_down),
                ):
                    with self.assertRaises(InferenceUnavailable):
                        submit_inference(app, self.request)
                self.assertFalse(app.state.admission_guard.locked())
                self.assertEqual(app.state.producer_futures, set())

    async def test_busy_rejection_and_consumer_cancellation_keep_worker_owned(self):
        started = Event()
        release = Event()
        loop_thread = get_ident()

        def held_generation(*args, **kwargs):
            self.assertNotEqual(get_ident(), loop_thread)
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test producer was not released.")
            return self.result

        with patch("src.api.generate", side_effect=held_generation):
            result = submit_inference(app, self.request)
            producer = next(iter(app.state.producer_futures))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                with self.assertRaises(InferenceBusy):
                    submit_inference(app, self.request)
                self.assertEqual(app.state.producer_futures, {producer})
                health = await self.client.get("/health")
                ready = await self.client.get("/ready")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(ready.status_code, 503)
                self.assertEqual(ready.json(), {"ready": False, "status": "busy"})

                consumer_started = asyncio.Event()

                async def consume():
                    consumer_started.set()
                    return await result

                consumer = asyncio.create_task(consume())
                await consumer_started.wait()
                consumer.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await consumer
                self.assertTrue(result.cancelled())
                self.assertFalse(producer.cancelled())
                self.assertFalse(producer.done())
                self.assertTrue(app.state.admission_guard.locked())
                self.assertEqual(app.state.producer_futures, {producer})
                with self.assertRaises(InferenceBusy):
                    submit_inference(app, self.request)
            finally:
                release.set()
                await asyncio.wait_for(asyncio.shield(producer), timeout=3)

            self.assertFalse(app.state.admission_guard.locked())
            self.assertEqual(app.state.producer_futures, set())
            ready = await self.client.get("/ready")
            self.assertEqual(ready.status_code, 200)
            recovered = await submit_inference(app, self.request)
            self.assertEqual(recovered, self.result)

    async def test_context_and_generation_failures_release_slot_for_recovery(self):
        for failure in (ContextLimitError("Oversized request"), RuntimeError("Failed")):
            with self.subTest(failure=type(failure).__name__):
                with patch("src.api.generate", side_effect=[failure, self.result]):
                    with self.assertRaises(type(failure)):
                        await submit_inference(app, self.request)
                    self.assertFalse(app.state.admission_guard.locked())
                    self.assertEqual(app.state.producer_futures, set())
                    recovered = await submit_inference(app, self.request)
                    self.assertEqual(recovered, self.result)

    async def test_submission_failure_releases_slot_for_recovery(self):
        with patch.object(app.state.executor, "submit", side_effect=RuntimeError("No worker")):
            with self.assertRaises(RuntimeError):
                submit_inference(app, self.request)
        self.assertFalse(app.state.admission_guard.locked())
        self.assertEqual(app.state.producer_futures, set())
        with patch("src.api.generate", return_value=self.result):
            recovered = await submit_inference(app, self.request)
        self.assertEqual(recovered, self.result)

    async def test_abandoned_failure_is_observed_and_does_not_leak_busy_slot(self):
        started = Event()
        release = Event()
        completed = asyncio.Event()
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        unhandled = []
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))

        def failing_generation(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test producer was not released.")
            raise RuntimeError("Private failure details")

        try:
            with patch("src.api.generate", side_effect=failing_generation):
                result = submit_inference(app, self.request)
                producer = next(iter(app.state.producer_futures))
                producer.add_done_callback(lambda future: completed.set())
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 2))
                    result.cancel()
                    with self.assertRaises(InferenceBusy):
                        submit_inference(app, self.request)
                finally:
                    release.set()
                    await asyncio.wait_for(completed.wait(), timeout=3)
                self.assertFalse(app.state.admission_guard.locked())
                self.assertEqual(app.state.producer_futures, set())
                # Never await or retrieve the producer exception in this test.
                del producer
                del result
            await asyncio.sleep(0)
            gc.collect()
            self.assertEqual(unhandled, [])
            with patch("src.api.generate", return_value=self.result):
                recovered = await submit_inference(app, self.request)
            self.assertEqual(recovered, self.result)
        finally:
            release.set()
            loop.set_exception_handler(previous_handler)


class CompletionEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        loader_patch = patch("src.api.load_runtime", return_value=object())
        loader_patch.start()
        self.addCleanup(loader_patch.stop)
        context = app.router.lifespan_context(app)
        await context.__aenter__()
        self.addAsyncCleanup(context.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )
        self.addAsyncCleanup(self.client.aclose)
        self.path = "/v1/chat/completions"
        self.result = GenerationResult("answer", 3, [7, 2], "stop")

    def assert_error(self, response, status, code):
        self.assertEqual(response.status_code, status)
        error = response.json()["error"]
        self.assertEqual(error["code"], code)
        request_id = response.headers["X-Request-ID"]
        self.assertEqual(UUID(request_id).hex, request_id)
        self.assertEqual(error["request_id"], request_id)

    async def test_validation_errors_are_private_and_precede_busy_admission(self):
        marker = "PRIVATE-INPUT-MARKER"
        cases = [
            (b'{"messages": "PRIVATE-INPUT-MARKER",', "invalid_request"),
            (b"\xff", "invalid_request"),
            (b"[]", "invalid_request"),
            ({}, "invalid_request"),
            (request_data(model=marker), "invalid_request"),
            (request_data(messages=[]), "invalid_request"),
            (request_data(messages=[{"role": marker, "content": marker}]), "invalid_request"),
            (request_data(messages=[{"role": "user", "content": " \t\n"}]), "invalid_request"),
            (request_data(max_tokens=-1), "invalid_request"),
            (request_data(max_tokens=marker), "invalid_request"),
            (request_data(temperature=True), "invalid_request"),
            (request_data(temperature=float("nan")), "invalid_request"),
            (request_data(**{marker: marker}), "invalid_request"),
            (request_data(messages=[{"role": "user", "content": marker, marker: marker}]), "invalid_request"),
            (request_data(top_p=0.9), "invalid_parameter_combination"),
            (request_data(stream=True), "unsupported_parameter"),
        ]
        app.state.admission_guard.acquire()
        try:
            for body, code in cases:
                with self.subTest(body=body):
                    response = await self.client.post(
                        self.path,
                        content=body if isinstance(body, bytes) else json.dumps(body),
                        headers={
                            "Content-Type": "application/json",
                            "X-Request-ID": marker,
                        },
                    )
                    self.assert_error(response, 422, code)
                    self.assertNotIn(marker, response.text)
                    self.assertNotEqual(response.headers["X-Request-ID"], marker)
            self.assertEqual(app.state.producer_futures, set())
        finally:
            app.state.admission_guard.release()

    async def test_unavailable_application_returns_503(self):
        for initialized, shutting_down in ((False, False), (True, True)):
            with self.subTest(initialized=initialized, shutting_down=shutting_down):
                with (
                    patch.object(app.state, "initialized", initialized),
                    patch.object(app.state, "shutting_down", shutting_down),
                ):
                    response = await self.client.post(self.path, json=request_data())
                self.assert_error(response, 503, "not_ready")
                self.assertFalse(app.state.admission_guard.locked())

    async def test_preparation_and_generation_errors_allow_later_success(self):
        marker = "PRIVATE-INPUT-MARKER"
        for failure, status, code in (
            (
                ContextLimitError(
                    "Prompt tokens (2000) plus requested output tokens (100) "
                    "exceed the supported limit (2048)."
                ),
                400,
                "context_length_exceeded",
            ),
            (RuntimeError(marker), 500, "generation_failed"),
        ):
            with self.subTest(code=code):
                with patch("src.api.generate", side_effect=[failure, self.result]):
                    response = await self.client.post(
                        self.path,
                        json=request_data(messages=[{"role": "user", "content": marker}]),
                    )
                    self.assert_error(response, status, code)
                    self.assertNotIn(marker, response.text)
                    self.assertFalse(app.state.admission_guard.locked())
                    recovered = await self.client.post(self.path, json=request_data())
                    self.assertEqual(recovered.status_code, 200)
                ready = await self.client.get("/ready")
                self.assertEqual(ready.status_code, 200)

    async def test_submission_failure_returns_safe_500_and_releases_slot(self):
        with patch.object(app.state.executor, "submit", side_effect=RuntimeError("SECRET")):
            response = await self.client.post(self.path, json=request_data())
        self.assert_error(response, 500, "generation_failed")
        self.assertNotIn("SECRET", response.text)
        self.assertFalse(app.state.admission_guard.locked())
        with patch("src.api.generate", return_value=self.result):
            recovered = await self.client.post(self.path, json=request_data())
        self.assertEqual(recovered.status_code, 200)

    async def test_busy_and_canceled_http_handler_retain_producer_ownership(self):
        started = Event()
        release = Event()

        def held_generation(*args, **kwargs):
            started.set()
            if not release.wait(5):
                raise TimeoutError("Test producer was not released.")
            return self.result

        with patch("src.api.generate", side_effect=held_generation):
            first = asyncio.create_task(self.client.post(self.path, json=request_data()))
            producer = None
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 2))
                producer = next(iter(app.state.producer_futures))
                health = await self.client.get("/health")
                ready = await self.client.get("/ready")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(ready.status_code, 503)
                second = await self.client.post(self.path, json=request_data())
                self.assert_error(second, 429, "server_busy")
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                self.assertFalse(producer.done())
                self.assertTrue(app.state.admission_guard.locked())
                third = await self.client.post(self.path, json=request_data())
                self.assert_error(third, 429, "server_busy")
            finally:
                release.set()
                if producer is not None:
                    await asyncio.wait_for(asyncio.shield(producer), timeout=3)
                await asyncio.gather(first, return_exceptions=True)
            recovered = await self.client.post(self.path, json=request_data())
            self.assertEqual(recovered.status_code, 200)
            self.assertFalse(app.state.admission_guard.locked())

    async def test_unknown_routes_and_wrong_methods_keep_normal_routing(self):
        missing = await self.client.post("/not-a-route", json=request_data())
        wrong_method = await self.client.get(self.path)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(wrong_method.status_code, 405)
        self.assertIn("POST", wrong_method.headers["allow"])
        self.assertNotIn("error", missing.json())
        self.assertNotIn("error", wrong_method.json())


if __name__ == "__main__":
    unittest.main()
