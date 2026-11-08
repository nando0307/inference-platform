"""Opt-in, single-resident vLLM routing with bounded FIFO admission."""

import asyncio
import json
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from time import perf_counter, time
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from pydantic import Field, ValidationError, field_validator
from starlette.requests import ClientDisconnect
from starlette.responses import JSONResponse, StreamingResponse

from .api import ChatCompletionRequest, ChatMessage, _DisconnectedResponse
from .inference import ContextLimitError
from .vllm_engine import MODEL_SPECS, VLLMEngine


MAX_BODY_BYTES = 65536
MAX_PROMPT_CHARS = 16384
MAX_RESPONSE_BYTES = 1024 * 1024
SEND_TIMEOUT = 5.0
DRAIN_TIMEOUT = 5.0
CLOSE_TIMEOUT = 3.0
STOP_TIMEOUT = 20.0


class RoutingRequest(ChatCompletionRequest):
    messages: list[ChatMessage] = Field(min_length=1, max_length=64)

    @field_validator("model")
    @classmethod
    def require_supported_model(cls, value):
        # Overriding the same validator name replaces the CPU-only allowlist.
        if value not in MODEL_SPECS:
            raise ValueError("Unsupported model.")
        return value

    @field_validator("messages")
    @classmethod
    def bound_prompt(cls, value):
        if sum(len(message.content) for message in value) > MAX_PROMPT_CHARS:
            raise ValueError("Prompt is too large.")
        return value


class RouterError(Exception):
    def __init__(self, status=503, code="not_ready", message="Inference is unavailable."):
        self.status, self.code, self.message = status, code, message
        super().__init__(code)


def _error(error, request_id):
    return {"error": {"code": error.code, "message": error.message, "request_id": request_id}}


def _observe(future):
    if not future.cancelled():
        future.exception()


@dataclass(eq=False)
class Lease:
    request_id: str
    request: RoutingRequest
    ready: asyncio.Future
    prepared: dict | None = None
    task: asyncio.Task | None = None
    upstream: httpx.Response | None = None
    dispatched: bool = False


class ModelManager:
    """All state and process transitions belong to one event loop and one pump."""

    def __init__(self, engine):
        self.engine = engine
        self.state = "unloaded"
        self.pending_model = None
        self.accepting = True
        self.queue = deque()
        self.active = set()
        self.events = deque(maxlen=2048)
        self.wake = asyncio.Event()
        self.recycle = False
        self.closing = False
        self.pump = None
        self.cleanup_error = None

    def snapshot(self):
        accepting = self.accepting and len(self.queue) < 2
        return {
            "state": self.state, "model": self.engine.model,
            "pending_model": self.pending_model, "active": len(self.active),
            "queued": len(self.queue), "accepting": accepting,
            "ready": accepting and self.state == "ready" and self.engine.is_alive(),
        }

    def emit(self, event, lease=None, **details):
        snapshot = self.snapshot()
        row = {"timestamp": perf_counter(), "event": event, **snapshot,
               "resident_model": snapshot["model"], **details}
        if lease is not None:
            row.update(request_id=lease.request_id, model=lease.request.model)
        self.events.append(row)

    def start(self):
        self.pump = asyncio.create_task(self._run(), name="model-router-lifecycle")

    def submit(self, body, request_id):
        if not self.accepting:
            raise RouterError()
        if len(self.queue) >= 2:
            raise RouterError(429, "server_busy", "The inference request queue is full.")
        lease = Lease(request_id, body, asyncio.get_running_loop().create_future())
        lease.ready.add_done_callback(_observe)
        self.queue.append(lease)
        self.emit("queued", lease)
        self.wake.set()
        return lease

    def cancel_waiting(self, lease):
        if lease in self.queue:
            self.queue.remove(lease)
            lease.ready.cancel()
            self.emit("canceled", lease, reason="client_disconnected")
            self.wake.set()

    def fail(self, reason="engine_failed", exclude=None):
        if self.state == "failed":
            return
        self.accepting = False
        self.state = "failed"
        self.pending_model = None
        self.emit("failed", reason=reason)
        while self.queue:
            lease = self.queue.popleft()
            if not lease.ready.done():
                lease.ready.set_exception(RouterError(503, reason, "The inference engine failed."))
        for lease in self.active:
            if lease is not exclude and lease.task is not None and lease.task is not asyncio.current_task():
                lease.task.cancel()
        self.wake.set()

    async def release(self, lease, completed):
        self.cancel_waiting(lease)
        if lease not in self.active:
            return
        if lease.upstream is not None:
            try:
                async with asyncio.timeout(CLOSE_TIMEOUT):
                    await lease.upstream.aclose()
            except Exception:
                completed = False
                self.fail("upstream_close_failed")
        self.active.remove(lease)
        if not completed and lease.dispatched:
            # HTTP cancellation has no abort acknowledgement. Drain other leases,
            # then stop the process group before admitting any further generation.
            self.recycle = True
        self.emit("released" if completed else "canceled", lease)
        self.wake.set()

    async def _stop(self):
        model = self.engine.model
        if self.state != "failed":
            self.state = "stopping"
        self.emit("stop_started", model=model)
        try:
            async with asyncio.timeout(STOP_TIMEOUT):
                await self.engine.stop()
        except Exception as error:
            self.cleanup_error = error
            raise
        self.emit("stop_finished", model=model)
        self.recycle = False
        if self.state != "failed":
            self.state = "unloaded"

    async def _step(self):
        if self.engine.model is not None and not self.engine.is_alive():
            self.fail()
        if self.state == "failed" or self.closing:
            if not self.active:
                return False
            return True
        if self.recycle:
            self.state = "draining"
            if self.active:
                return True
            await self._stop()
        if not self.queue:
            if self.engine.model is not None:
                self.state = "ready"
            self.pending_model = None
            return True
        lease = self.queue[0]
        model = lease.request.model
        self.pending_model = model if model != self.engine.model else None
        if lease.prepared is None:
            try:
                lease.prepared = await self.engine.prepare(
                    model, [message.model_dump() for message in lease.request.messages],
                    lease.request.max_tokens,
                )
            except ContextLimitError:
                if lease in self.queue:
                    self.queue.remove(lease)
                    lease.ready.set_exception(RouterError(
                        400, "context_length_exceeded", "Prompt and output exceed the supported context.",
                    ))
                self.wake.set()
                return True
            # Preparation may yield while a waiting client cancels.
            if lease not in self.queue or self.closing:
                return True
        if self.engine.model != model:
            self.state = "draining" if self.active else "loading"
            if self.active:
                return True
            if self.engine.model is not None:
                await self._stop()
            self.state = "loading"
            self.emit("load_started", model=model)
            await self.engine.start(model)
            if self.engine.model != model or not self.engine.is_alive():
                raise RuntimeError("Engine did not start the requested model.")
            self.state = "ready"
            self.pending_model = None
            self.emit("load_finished", model=model)
        if lease not in self.queue or self.closing:
            return True
        if len(self.active) < 2:
            self.queue.popleft()
            self.active.add(lease)
            lease.ready.set_result(lease)
            self.emit("admitted", lease)
            self.wake.set()
        return True

    async def _run(self):
        try:
            while True:
                self.wake.clear()
                if not await self._step():
                    break
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=0.1)
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            pass
        except Exception:
            self.fail("load_failed" if self.state == "loading" else "engine_failed")
        finally:
            if self.active:
                try:
                    async with asyncio.timeout(DRAIN_TIMEOUT):
                        while self.active:
                            self.wake.clear()
                            await self.wake.wait()
                except TimeoutError:
                    pass
            try:
                await self._stop()
            except Exception as error:
                self.cleanup_error = error
                self.fail("cleanup_failed")

    async def shutdown(self):
        self.accepting = False
        self.closing = True
        self.emit("shutdown")
        while self.queue:
            lease = self.queue.popleft()
            lease.ready.set_exception(RouterError())
        self.wake.set()
        try:
            await asyncio.wait_for(asyncio.shield(self.pump), DRAIN_TIMEOUT)
        except TimeoutError:
            for lease in list(self.active):
                if lease.task is not None:
                    lease.task.cancel()
            # The lifecycle pump remains the only owner of process cleanup.
            self.pump.cancel()
            await self.pump
        if self.cleanup_error is not None:
            raise RuntimeError("Engine cleanup could not be confirmed.") from self.cleanup_error


class _LeaseMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state = scope.setdefault("state", {})
        state["request_id"] = uuid4().hex
        success = False

        async def bounded_send(message):
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []),
                                      (b"x-request-id", state["request_id"].encode())]
            async with asyncio.timeout(SEND_TIMEOUT):
                await send(message)

        try:
            await self.app(scope, receive, bounded_send)
            success = True
        finally:
            lease = state.get("lease")
            if lease is not None:
                cleanup = asyncio.create_task(scope["app"].state.manager.release(
                    lease, success and state.get("completed", False),
                ))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise


async def _connected(request, future):
    while not future.done():
        if await request.is_disconnected():
            raise ClientDisconnect()
        if asyncio.current_task().cancelling():
            raise asyncio.CancelledError()
        await asyncio.wait({future}, timeout=0.05)
    if await request.is_disconnected():
        raise ClientDisconnect()
    if asyncio.current_task().cancelling():
        raise asyncio.CancelledError()
    return future.result()


def _usage(value, lease, finish):
    fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(value, dict) or any(type(value.get(key)) is not int for key in fields):
        raise ValueError("Invalid usage.")
    prompt, output, total = (value[key] for key in fields)
    if (prompt != lease.prepared["prompt_tokens"] or not 0 <= output <= lease.request.max_tokens
            or total != prompt + output or (finish == "length" and output != lease.request.max_tokens)):
        raise ValueError("Invalid usage.")
    return {key: value[key] for key in fields}


def _identity(payload, lease, *, require_prompt=False):
    if not isinstance(payload, dict) or payload.get("model") != lease.request.model:
        raise ValueError("Wrong served model.")
    if "error" in payload or payload.get("id") != "chatcmpl-" + lease.request_id:
        raise ValueError("Invalid upstream identity.")
    ids = payload.get("prompt_token_ids")
    if require_prompt and ids is None:
        raise ValueError("Missing upstream prompt tokenization.")
    if ids is not None and (
        not isinstance(ids, list) or any(type(token) is not int for token in ids)
        or ids != lease.prepared["prompt_token_ids"]
    ):
        raise ValueError("Wrong prompt tokenization.")


def _choice(choice, streaming):
    if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
        raise ValueError("Invalid choice.")
    field = "delta" if streaming else "message"
    text = choice.get(field)
    if (not isinstance(text, dict) or text.get("role", "assistant") != "assistant"
            or (text.get("content") is not None and not isinstance(text["content"], str))
            or text.get("reasoning_content") or text.get("reasoning") or text.get("tool_calls")):
        raise ValueError("Invalid completion.")
    finish = choice.get("finish_reason")
    if finish not in (None, "stop", "length"):
        raise ValueError("Invalid finish reason.")
    result = {"index": 0, field: {key: value for key, value in text.items() if key in ("role", "content")},
              "finish_reason": finish}
    ids = choice.get("token_ids")
    if ids is not None:
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Invalid token IDs.")
        result["token_ids"] = ids
    return result


async def _upstream_events(response):
    pending = b""
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > MAX_RESPONSE_BYTES:
            raise ValueError("Oversized upstream response.")
        pending += chunk
        while b"\n\n" in pending or b"\r\n\r\n" in pending:
            windows = pending.find(b"\r\n\r\n")
            unix = pending.find(b"\n\n")
            end, width = (windows, 4) if windows >= 0 and (unix < 0 or windows < unix) else (unix, 2)
            frame, pending = pending[:end], pending[end + width:]
            if len(frame) > MAX_BODY_BYTES:
                raise ValueError("Oversized SSE event.")
            data = [line[5:].removeprefix(b" ") for line in frame.splitlines() if line.startswith(b"data:")]
            if data:
                yield b"\n".join(data).decode("utf-8")
        if len(pending) > MAX_BODY_BYTES:
            raise ValueError("Oversized SSE event.")
    if pending.strip():
        raise ValueError("Truncated SSE event.")


def _sse(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


async def _stream(request, lease):
    finish = None
    usage = None
    done = False
    first = True
    try:
        async with asyncio.timeout(180):
            async for event in _upstream_events(lease.upstream):
                if done:
                    raise ValueError("Data after terminal marker.")
                if event == "[DONE]":
                    if finish is None or usage is None:
                        raise ValueError("Incomplete stream.")
                    done = True
                    continue
                payload = json.loads(event)
                _identity(payload, lease, require_prompt=first)
                first = False
                choices = payload.get("choices")
                if not isinstance(choices, list) or len(choices) > 1:
                    raise ValueError("Invalid choices.")
                if choices:
                    if finish is not None:
                        raise ValueError("Data after finish.")
                    payload["choices"] = [_choice(choices[0], True)]
                    finish = payload["choices"][0]["finish_reason"]
                elif finish is None or payload.get("usage") is None:
                    raise ValueError("Invalid terminal chunk.")
                if payload.get("usage") is not None:
                    if finish is None or usage is not None:
                        raise ValueError("Invalid usage ordering.")
                    usage = _usage(payload["usage"], lease, finish)
                    payload["usage"] = usage
                yield _sse({key: value for key, value in payload.items() if key in (
                    "id", "object", "created", "model", "choices", "usage", "prompt_token_ids",
                )})
        if not done:
            raise ValueError("Missing terminal marker.")
        yield "data: [DONE]\n\n"
        request.state.completed = True
    except asyncio.CancelledError:
        raise
    except Exception:
        request.app.state.manager.fail("upstream_failed", exclude=lease)
        yield _sse(_error(RouterError(502, "upstream_failed", "The inference upstream failed."), lease.request_id))
        yield "data: [DONE]\n\n"


class _ClosingStream(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.body_iterator.aclose()


async def _complete(request, lease):
    body = lease.request
    payload = {
        **body.model_dump(), "n": 1, "seed": 0, "top_k": -1, "min_p": 0,
        "presence_penalty": 0, "frequency_penalty": 0, "repetition_penalty": 1,
        "ignore_eos": False, "min_tokens": 0, "stop_token_ids": lease.prepared["eos_token_ids"],
        "skip_special_tokens": True, "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if body.stream:
        payload["stream_options"] = {"include_usage": True}
    client = request.app.state.client
    upstream = client.build_request(
        "POST", request.app.state.manager.engine.base_url.rstrip("/") + "/v1/chat/completions",
        json=payload, headers={"x-request-id": lease.request_id},
    )
    lease.dispatched = True
    lease.upstream = await client.send(upstream, stream=True)
    if lease.upstream.status_code != 200:
        raise ValueError("Upstream rejected completion.")
    if body.stream:
        if lease.upstream.headers.get("content-type", "").split(";", 1)[0].strip() != "text/event-stream":
            raise ValueError("Invalid upstream content type.")
        return _ClosingStream(_stream(request, lease), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache"})
    raw = bytearray()
    async for chunk in lease.upstream.aiter_bytes():
        if len(raw) + len(chunk) > MAX_RESPONSE_BYTES:
            raise ValueError("Oversized upstream response.")
        raw.extend(chunk)
    payload = json.loads(raw)
    _identity(payload, lease, require_prompt=True)
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Invalid choices.")
    choice = _choice(choices[0], False)
    if choice["finish_reason"] is None or not isinstance(choice["message"].get("content"), str):
        raise ValueError("Incomplete completion.")
    usage = _usage(payload.get("usage"), lease, choice["finish_reason"])
    request.state.completed = True
    return JSONResponse({"id": payload["id"], "object": "chat.completion", "created": int(time()),
                         "model": body.model, "choices": [choice], "usage": usage,
                         "prompt_token_ids": payload["prompt_token_ids"]})


def create_app(engine=None):
    engine = engine if engine is not None else VLLMEngine()

    @asynccontextmanager
    async def lifespan(app):
        manager = ModelManager(engine)
        app.state.manager = manager
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=120, write=10, pool=1),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=2),
            trust_env=False, follow_redirects=False,
        ) as client:
            app.state.client = client
            manager.start()
            try:
                yield
            finally:
                await manager.shutdown()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(_LeaseMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "alive"}

    @app.get("/ready")
    async def ready():
        state = app.state.manager.snapshot()
        return JSONResponse(state, status_code=200 if state["ready"] else 503)

    @app.post("/v1/chat/completions")
    async def completion(request: Request):
        lease = None
        work = None
        try:
            raw = bytearray()
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > MAX_BODY_BYTES:
                        raise RouterError(413, "request_too_large", "Request body is too large.")
                    raw.extend(chunk)
            try:
                body = RoutingRequest.model_validate_json(raw)
            except (ValidationError, ValueError):
                raise RouterError(422, "invalid_request", "Request does not match the supported schema.") from None
            lease = app.state.manager.submit(body, request.state.request_id)
            request.state.lease = lease
            lease.task = asyncio.current_task()
            await _connected(request, lease.ready)
            work = asyncio.create_task(_complete(request, lease))
            async with asyncio.timeout(180):
                return await _connected(request, work)
        except ClientDisconnect:
            return _DisconnectedResponse()
        except RouterError as error:
            return JSONResponse(_error(error, request.state.request_id), status_code=error.status)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            error = RouterError(504, "request_timeout", "The request exceeded its deadline.")
            return JSONResponse(_error(error, request.state.request_id), status_code=error.status)
        except Exception:
            app.state.manager.fail("upstream_failed")
            error = RouterError(502, "upstream_failed", "The inference upstream failed.")
            return JSONResponse(_error(error, request.state.request_id), status_code=error.status)
        finally:
            if work is not None and not work.done():
                work.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await work

    return app
