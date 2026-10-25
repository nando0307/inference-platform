import asyncio
import json
import logging
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Empty
from time import perf_counter, time
from typing import Literal, Self
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Receive, Scope, Send

from .inference import (
    MODEL_ID,
    CancellationToken,
    ContextLimitError,
    GenerationCancelled,
    GenerationResult,
    Runtime,
    generate,
    load_runtime,
)
from .streaming import BoundedTextIteratorStreamer


QUEUE_CAPACITY = 2
STREAM_SEND_TIMEOUT = 5.0


logger = logging.getLogger("inference.lifecycle")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


@dataclass
class _RequestLog:
    """Event-loop-owned metadata only; never retain bodies, results, or exceptions."""

    started: float
    request_id: str = field(default_factory=lambda: uuid4().hex)
    stream: bool | None = None
    http_status: int | None = None
    outcome: str = "completed"
    error_code: str | None = None
    exception_type: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    queue_wait_ms: float | None = None
    cancellation_delay_ms: float | None = None
    finished: bool = False

    def emit(self, event: str, **fields) -> None:
        logger.info(json.dumps({
            "event": event,
            "request_id": self.request_id,
            "model": MODEL_ID,
            "stream": self.stream,
            "http_status": self.http_status,
            **fields,
        }))

    def error(self, code: str, error: BaseException | None = None) -> None:
        self.outcome = (
            "failed" if code in {"generation_failed", "slow_consumer", "request_canceled"}
            else "rejected"
        )
        self.error_code = code
        self.exception_type = type(error).__name__ if error is not None else None

    def finish(self, outcome: str | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        fields = {
            "outcome": outcome or self.outcome,
            "duration_ms": (perf_counter() - self.started) * 1000,
        }
        if self.error_code is not None:
            fields["error_code"] = self.error_code
        if self.exception_type is not None:
            fields["exception_type"] = self.exception_type
        if self.prompt_tokens is not None:
            fields["prompt_tokens"] = self.prompt_tokens
            fields["completion_tokens"] = self.completion_tokens
        if self.queue_wait_ms is not None:
            fields["queue_wait_ms"] = self.queue_wait_ms
        if self.cancellation_delay_ms is not None:
            fields["cancellation_delay_ms"] = self.cancellation_delay_ms
        self.emit("request_finished", **fields)

    def scheduling_finished(
        self, job: "_InferenceJob", result: GenerationResult | None,
        error: BaseException | None,
    ) -> None:
        self.queue_wait_ms = (
            (job.started_at - job.admitted_at) * 1000
            if job.started_at is not None else None
        )
        requested_at = job.cancellation.requested_at
        if requested_at is not None and requested_at > job.finished_at:
            requested_at = None
        self.cancellation_delay_ms = (
            (job.finished_at - requested_at) * 1000
            if job.started_at is not None and requested_at is not None else None
        )
        metrics = {
            "queue_wait_ms": self.queue_wait_ms,
            "cancellation_delay_ms": self.cancellation_delay_ms,
        }
        if requested_at is not None:
            metrics["cancellation_reason"] = job.cancellation.reason
        if job.started_at is not None:
            fields = {
                **metrics,
                "outcome": "completed",
                "producer_duration_ms": (job.finished_at - job.started_at) * 1000,
            }
            if result is not None:
                self.prompt_tokens = result.prompt_tokens
                self.completion_tokens = len(result.generated_ids)
                fields.update(
                    prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                )
            if error is not None:
                _, code, _ = _generation_error(error)
                fields["outcome"] = (
                    "canceled" if isinstance(error, GenerationCancelled)
                    else "rejected" if isinstance(error, ContextLimitError) else "failed"
                )
                fields["error_code"] = code
                if not isinstance(error, (GenerationCancelled, ContextLimitError)):
                    fields["exception_type"] = type(error).__name__
            self.emit("producer_finished", **fields)
        self.emit("scheduling_finished", state=job.state, **metrics)


class _RequestLoggingMiddleware:
    """Observe ASGI delivery boundaries without reading or buffering message bodies."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] != "/v1/chat/completions"
        ):
            await self.app(scope, receive, send)
            return

        request_log = _RequestLog(perf_counter())
        scope_state = scope.setdefault("state", {})
        scope_state["request_log"] = request_log

        def cancel_job(reason):
            job = scope_state.get("inference_job")
            if job is not None:
                cancel_inference(scope["app"], job, reason)

        async def receive_logged():
            message = await receive()
            if message["type"] == "http.disconnect":
                scope_state["disconnected"] = True
                cancel_job("client_disconnected")
                request_log.finish("disconnected")
            return message

        async def send_logged(message):
            if message["type"] == "http.response.start":
                request_log.http_status = message["status"]
                message = {
                    **message,
                    "headers": [
                        (name, value) for name, value in message.get("headers", [])
                        if name.lower() != b"x-request-id"
                    ] + [(b"x-request-id", request_log.request_id.encode("ascii"))],
                }
            try:
                await asyncio.wait_for(send(message), timeout=STREAM_SEND_TIMEOUT)
            except TimeoutError:
                cancel_job("slow_consumer")
                request_log.error("slow_consumer")
                request_log.finish("failed")
                # A transport that cannot send is not a place to send an error frame.
                raise OSError("Chat transport send deadline exceeded.") from None
            except OSError:
                cancel_job("client_disconnected")
                request_log.finish("disconnected")
                raise ClientDisconnect() from None
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                request_log.finish()

        try:
            await self.app(scope, receive_logged, send_logged)
        except (ClientDisconnect, OSError):
            cancel_job("client_disconnected")
            request_log.finish("disconnected")
        except asyncio.CancelledError:
            cancel_job("request_canceled")
            request_log.finish("disconnected")
            raise
        except Exception as error:
            cancel_job("request_canceled")
            request_log.error("generation_failed", error)
            if request_log.http_status is None:
                response = _error_response(
                    500, "generation_failed", "The completion could not be generated.",
                    request_log, error,
                )
                await response(scope, receive_logged, send_logged)
            else:
                # Do not let Uvicorn print the original exception's private message.
                raise RuntimeError("Chat response failed after headers.") from None
        finally:
            if not request_log.finished:
                request_log.error_code = request_log.error_code or "generation_failed"
                request_log.finish("failed")


class ChatMessage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def require_nonblank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must contain non-whitespace text.")
        return value


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    stream: bool = False

    @field_validator("model")
    @classmethod
    def require_supported_model(cls, value: str) -> str:
        if value != MODEL_ID:
            raise ValueError(f"The supported model is {MODEL_ID}.")
        return value

    @model_validator(mode="after")
    def require_supported_settings(self) -> Self:
        if self.temperature == 0 and self.top_p != 1:
            raise PydanticCustomError(
                "invalid_parameter_combination",
                "Greedy decoding requires top_p=1.0.",
            )
        return self


class InferenceUnavailable(RuntimeError):
    """The application is not accepting inference work."""


class InferenceBusy(RuntimeError):
    """Both waiting slots are occupied."""


@dataclass(eq=False)
class _InferenceJob:
    request: ChatCompletionRequest
    request_log: _RequestLog | None
    streamer: BoundedTextIteratorStreamer | None
    prepared: asyncio.Future[None] | None
    result: asyncio.Future[GenerationResult]
    cancellation: CancellationToken
    admitted_at: float
    state: str = "queued"
    started_at: float | None = None
    finished_at: float | None = None
    producer: asyncio.Future[GenerationResult] | None = None


def _observe_result(future: asyncio.Future) -> None:
    if not future.cancelled():
        future.exception()


def submit_inference(
    app: FastAPI,
    request: ChatCompletionRequest,
    *,
    streamer: BoundedTextIteratorStreamer | None = None,
    prepared: asyncio.Future[None] | None = None,
    request_log: _RequestLog | None = None,
) -> _InferenceJob:
    """Admit on the event loop; waiting jobs never enter the executor."""
    loop = asyncio.get_running_loop()
    state = app.state
    if not getattr(state, "initialized", False) or state.shutting_down:
        raise InferenceUnavailable("Inference is not available.")
    if state.active_job is not None and len(state.pending_jobs) >= QUEUE_CAPACITY:
        raise InferenceBusy("The inference request queue is full.")
    admitted_at = perf_counter()
    cancellation = streamer.cancellation if streamer is not None else CancellationToken()
    if request.stream and streamer is None:
        streamer = BoundedTextIteratorStreamer(
            state.runtime.tokenizer, cancellation=cancellation,
            skip_prompt=True, skip_special_tokens=True,
            clean_up_tokenization_spaces=False, timeout=0.1,
        )
    result = loop.create_future()
    result.add_done_callback(_observe_result)
    if prepared is not None:
        prepared.add_done_callback(_observe_result)
    job = _InferenceJob(
        request, request_log, streamer, prepared, result, cancellation, admitted_at,
    )
    if state.active_job is None:
        job.state = "running"
    if request_log is not None:
        request_log.emit("request_accepted", state=job.state)
    if job.state == "running":
        _start_job(app, job)
    else:
        state.pending_jobs.append(job)
    return job


def _start_job(app: FastAPI, job: _InferenceJob) -> None:
    state = app.state
    state.active_job = job
    job.state = "running"
    loop = asyncio.get_running_loop()
    try:
        producer = loop.run_in_executor(state.executor, _run_inference, state.runtime, job, loop)
    except Exception as error:
        _finish_job(job, None, error)
        state.active_job = None
        _promote_next(app)
        return
    job.producer = producer
    state.producer_futures.add(producer)
    producer.add_done_callback(lambda future: _producer_done(app, job, future))


def _request_started(job: _InferenceJob) -> None:
    if job.request_log is not None:
        job.request_log.queue_wait_ms = (job.started_at - job.admitted_at) * 1000
        job.request_log.emit(
            "request_started", state="running", queue_wait_ms=job.request_log.queue_wait_ms,
        )


def _run_inference(runtime: Runtime, job: _InferenceJob, loop) -> GenerationResult:
    job.started_at = perf_counter()
    loop.call_soon_threadsafe(_request_started, job)

    def on_prepared():
        loop.call_soon_threadsafe(_resolve_preparation, job.prepared, None)

    try:
        job.cancellation.raise_if_cancelled()
        result = generate(
            runtime, [message.model_dump() for message in job.request.messages],
            max_new_tokens=job.request.max_tokens,
            temperature=job.request.temperature, top_p=job.request.top_p,
            streamer=job.streamer,
            on_prepared=on_prepared if job.prepared is not None else None,
            cancellation=job.cancellation,
        )
        job.cancellation.raise_if_cancelled()
        return result
    finally:
        job.finished_at = perf_counter()


def _resolve_preparation(future: asyncio.Future[None] | None, error: BaseException | None):
    if future is not None and not future.done():
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)


def _finish_job(
    job: _InferenceJob, result: GenerationResult | None, error: BaseException | None,
) -> None:
    job.finished_at = job.finished_at if job.finished_at is not None else perf_counter()
    job.state = (
        "canceled" if isinstance(error, GenerationCancelled)
        else "failed" if error is not None else "completed"
    )
    _resolve_preparation(job.prepared, error)
    if not job.result.done():
        if error is None:
            job.result.set_result(result)
        else:
            job.result.set_exception(error)
    if job.request_log is not None:
        job.request_log.scheduling_finished(job, result, error)


def _producer_done(app: FastAPI, job: _InferenceJob, producer: asyncio.Future) -> None:
    # Only actual worker completion releases ownership, never HTTP cancellation.
    app.state.producer_futures.discard(producer)
    error = producer.exception()
    # Worker exit can race the check in cancel_inference, before its token signal.
    requested_at = job.cancellation.requested_at
    if error is None and requested_at is not None and requested_at <= job.finished_at:
        error = GenerationCancelled(job.cancellation.reason)
    _finish_job(job, producer.result() if error is None else None, error)
    app.state.active_job = None
    _promote_next(app)


def _promote_next(app: FastAPI) -> None:
    state = app.state
    if not state.shutting_down and state.pending_jobs:
        _start_job(app, state.pending_jobs.popleft())


def cancel_inference(app: FastAPI, job: _InferenceJob, reason="client_disconnected") -> None:
    if job.state == "queued":
        app.state.pending_jobs.remove(job)
        job.cancellation.cancel(reason)
        _finish_job(job, None, GenerationCancelled(job.cancellation.reason))
    elif job.state == "running" and job.finished_at is None:
        job.cancellation.cancel(reason)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = app.state
    state.initialized = False
    state.shutting_down = False
    state.runtime = None
    state.executor = None
    state.active_job = None
    state.pending_jobs = deque()
    state.producer_futures = set()

    try:
        state.runtime = load_runtime()
        state.executor = ThreadPoolExecutor(max_workers=1)
        state.initialized = True
        yield
    finally:
        # Uvicorn normally drains HTTP first. This handles any jobs left at teardown.
        state.initialized = False
        state.shutting_down = True
        for job in tuple(state.pending_jobs):
            cancel_inference(app, job, "shutdown")
        if state.active_job is not None:
            cancel_inference(app, state.active_job, "shutdown")
        if state.executor is not None:
            await asyncio.to_thread(state.executor.shutdown, wait=True)
            state.executor = None
        if state.producer_futures:
            await asyncio.gather(
                *(asyncio.shield(future) for future in state.producer_futures),
                return_exceptions=True,
            )
        state.runtime = None


app = FastAPI(lifespan=lifespan)
app.add_middleware(_RequestLoggingMiddleware)


def _error_payload(code: str, message: str, request_id: str) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id}}


def _usage(result: GenerationResult) -> dict[str, int]:
    completion_tokens = len(result.generated_ids)
    return {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": result.prompt_tokens + completion_tokens,
    }


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_log: _RequestLog,
    error: BaseException | None = None,
) -> JSONResponse:
    request_log.error(code, error)
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(code, message, request_log.request_id),
    )


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, error: StarletteHTTPException):
    # FastAPI wraps invalid JSON encoding as a body-parsing HTTP 400.
    if (
        error.status_code == 400
        and isinstance(error.__cause__, UnicodeError)
        and request.method == "POST"
        and request.url.path == "/v1/chat/completions"
    ):
        return _error_response(
            422, "invalid_request", "Request body must contain valid JSON.",
            request.state.request_log,
        )
    if request_log := getattr(request.state, "request_log", None):
        request_log.error(
            "generation_failed" if error.status_code >= 500 else "invalid_request"
        )
    return await http_exception_handler(request, error)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    request_log = request.state.request_log
    # Extract only a genuine boolean, never arbitrary invalid values or field names.
    if isinstance(error.body, dict):
        stream = error.body.get("stream", False)
        if type(stream) is bool:
            request_log.stream = stream
    first = error.errors()[0]
    error_type = first["type"]
    if error_type == "invalid_parameter_combination":
        code = error_type
        message = "Greedy decoding requires top_p=1.0."
    else:
        code = "invalid_request"
        # Never serialize raw validation inputs, contexts, messages, or field names.
        if error_type == "json_invalid":
            message = "Request body must contain valid JSON."
        elif error_type == "extra_forbidden":
            message = "Unknown request or message fields are not supported."
        else:
            location = first.get("loc", ())
            field = location[1] if len(location) > 1 else None
            message = {
                "model": f"model must be {MODEL_ID}.",
                "messages": (
                    "messages must be a nonempty array with supported roles "
                    "(system, user, assistant) and nonblank string content."
                ),
                "max_tokens": "max_tokens must be an integer from 1 through 512.",
                "temperature": "temperature must be a finite number from 0 through 2.",
                "top_p": "top_p must be a finite number greater than 0 and at most 1.",
                "stream": "stream must be a boolean.",
            }.get(field, "Request body must be an object matching the supported schema.")
    return _error_response(422, code, message, request_log)


class _DisconnectedResponse(Response):
    """Finish the ASGI call without sending to a connection already known closed."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        return


class _ClosingStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # Starlette does not close an iterator suspended at a failed send.
            cancel_inference(scope["app"], scope["state"]["inference_job"], "request_canceled")
            await self.body_iterator.aclose()


def _generation_error(error: BaseException) -> tuple[int, str, str]:
    if isinstance(error, ContextLimitError):
        return 400, "context_length_exceeded", str(error)
    if isinstance(error, GenerationCancelled):
        if error.reason == "shutdown":
            return 503, "not_ready", "Inference is shutting down."
        if error.reason == "slow_consumer":
            return 503, "slow_consumer", "The client could not consume the completion in time."
        return 503, "request_canceled", "The completion was canceled."
    return 500, "generation_failed", "The completion could not be generated."


async def _is_disconnected(request: Request) -> bool:
    if getattr(request.state, "disconnected", False):
        return True
    disconnected = await request.is_disconnected()
    # Starlette's AnyIO probe can consume a concurrent Task.cancel(). Preserve it.
    if asyncio.current_task().cancelling():
        raise asyncio.CancelledError()
    return disconnected


async def _await_connected(request: Request, job: _InferenceJob, future: asyncio.Future):
    protected = asyncio.shield(future)
    protected.add_done_callback(_observe_result)
    try:
        while not protected.done():
            if await _is_disconnected(request):
                raise ClientDisconnect()
            await asyncio.wait({protected}, timeout=0.1)
        if await _is_disconnected(request):
            raise ClientDisconnect()
        return protected.result()
    except ClientDisconnect:
        cancel_inference(request.app, job, "client_disconnected")
        raise
    except asyncio.CancelledError:
        cancel_inference(request.app, job, "request_canceled")
        raise
    finally:
        protected.cancel()


def _next_text(streamer: BoundedTextIteratorStreamer) -> tuple[str, str]:
    # StopIteration must not cross an asyncio Future boundary.
    try:
        return "text", next(streamer)
    except Empty:
        return "timeout", ""
    except StopIteration:
        return "end", ""


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _stream_completion(
    request: Request,
    job: _InferenceJob,
    request_id: str,
    created: int,
):
    future = job.result
    base = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": created,
        "model": MODEL_ID,
    }

    def chunk(delta, finish_reason=None, usage=None):
        data = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            data["usage"] = usage
        return _sse(data)

    try:
        if await _is_disconnected(request):
            return
        yield chunk({"role": "assistant"})
        while True:
            if await _is_disconnected(request):
                return
            producer_done = future.done()
            if producer_done:
                future.result()  # Detect failure even without a streamer sentinel.
            outcome, text = await asyncio.to_thread(_next_text, job.streamer)
            if await _is_disconnected(request):
                return
            if outcome == "text":
                if text:
                    yield chunk({"content": text})
            elif outcome == "end":
                result = await _await_connected(request, job, future)
                break
            elif producer_done:
                # Work was complete before the empty queue read, so no text is lost.
                result = future.result()
                break
        yield chunk({}, result.finish_reason, _usage(result))
        yield "data: [DONE]\n\n"
    except ClientDisconnect:
        return
    except asyncio.CancelledError:
        cancel_inference(request.app, job, "request_canceled")
        raise
    except Exception as error:
        if isinstance(error, GenerationCancelled) and error.reason == "client_disconnected":
            return
        _, code, message = _generation_error(error)
        request.state.request_log.error(code, error)
        if await _is_disconnected(request):
            return
        yield _sse(_error_payload(code, message, request_id))
        yield "data: [DONE]\n\n"
    finally:
        reason = "client_disconnected" if getattr(request.state, "disconnected", False) else "request_canceled"
        cancel_inference(request.app, job, reason)


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    request_log = request.state.request_log
    request_log.stream = body.stream
    request_id = request_log.request_id
    created = int(time())
    try:
        prepared = asyncio.get_running_loop().create_future() if body.stream else None
        job = submit_inference(request.app, body, prepared=prepared, request_log=request_log)
        request.state.inference_job = job
        if body.stream:
            await _await_connected(request, job, prepared)
            if job.result.done():
                job.result.result()
            return _ClosingStreamingResponse(
                _stream_completion(request, job, request_id, created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        result = await _await_connected(request, job, job.result)
    except ClientDisconnect:
        return _DisconnectedResponse()
    except InferenceUnavailable:
        return _error_response(503, "not_ready", "Inference is not available.", request_log)
    except InferenceBusy:
        return _error_response(
            429, "server_busy", "The inference request queue is full.", request_log
        )
    except Exception as error:
        if isinstance(error, GenerationCancelled) and error.reason == "client_disconnected":
            return _DisconnectedResponse()
        status, code, message = _generation_error(error)
        logged_error = None if isinstance(error, (ContextLimitError, GenerationCancelled)) else error
        return _error_response(status, code, message, request_log, logged_error)

    return JSONResponse(
        content={
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": created,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": _usage(result),
        },
    )


@app.get("/health")
async def health():
    return {"status": "alive"}


@app.get("/ready")
async def ready(request: Request):
    state = request.app.state
    running = int(getattr(state, "active_job", None) is not None)
    queued = len(getattr(state, "pending_jobs", ()))
    if getattr(state, "shutting_down", False):
        status = "shutting_down"
    elif not getattr(state, "initialized", False):
        status = "not_ready"
    elif queued >= QUEUE_CAPACITY:
        status = "full"
    elif running:
        status = "busy"
    else:
        status = "ready"
    available = status in {"ready", "busy"}
    return JSONResponse(
        status_code=200 if available else 503,
        content={
            "ready": available, "status": status, "running": running,
            "queued": queued, "queue_capacity": QUEUE_CAPACITY,
        },
    )
