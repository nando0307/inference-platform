import asyncio
import json
import logging
from _thread import LockType
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from queue import Empty
from threading import Lock
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
from transformers import TextIteratorStreamer

from .inference import (
    MODEL_ID,
    ContextLimitError,
    GenerationResult,
    Runtime,
    generate,
    load_runtime,
)


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
        self.outcome = "failed" if code == "generation_failed" else "rejected"
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
        self.emit("request_finished", **fields)

    def producer_finished(
        self,
        outcome: str,
        duration_ms: float,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        error_code: str | None,
        exception_type: str | None,
    ) -> None:
        # Called on the event loop after worker exit, independently of delivery.
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        fields = {"outcome": outcome, "producer_duration_ms": duration_ms}
        if prompt_tokens is not None:
            fields["prompt_tokens"] = prompt_tokens
            fields["completion_tokens"] = completion_tokens
        if error_code is not None:
            fields["error_code"] = error_code
        if exception_type is not None:
            fields["exception_type"] = exception_type
        self.emit("producer_finished", **fields)


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
        scope.setdefault("state", {})["request_log"] = request_log

        async def receive_logged():
            message = await receive()
            if message["type"] == "http.disconnect":
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
                await send(message)
            except OSError:
                request_log.finish("disconnected")
                raise
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                request_log.finish()

        try:
            await self.app(scope, receive_logged, send_logged)
        except (ClientDisconnect, OSError):
            request_log.finish("disconnected")
        except asyncio.CancelledError:
            request_log.finish("disconnected")
            raise
        except Exception as error:
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
            # Normal completion was logged by send; disconnect by receive/cancellation.
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
    """Another producer owns the inference slot."""


def submit_inference(
    app: FastAPI,
    request: ChatCompletionRequest,
    *,
    streamer: TextIteratorStreamer | None = None,
    prepared: asyncio.Future[None] | None = None,
    request_log: _RequestLog | None = None,
) -> asyncio.Future[GenerationResult]:
    """Admit a validated request on the event loop and return a protected result."""
    loop = asyncio.get_running_loop()
    state = app.state
    if not getattr(state, "initialized", False) or state.shutting_down:
        raise InferenceUnavailable("Inference is not available.")
    if not state.admission_guard.acquire(blocking=False):
        raise InferenceBusy("Another inference request is running.")
    if prepared is not None:
        prepared.add_done_callback(lambda future: _observe_completion(app, future))

    try:
        producer = loop.run_in_executor(
            state.executor, _run_inference, state.runtime, request, state.admission_guard,
            loop, streamer, prepared, request_log,
        )
    except Exception as error:
        state.admission_guard.release()
        if prepared is not None:
            _resolve_preparation(prepared, error)
        raise

    state.producer_futures.add(producer)
    producer.add_done_callback(lambda future: _observe_completion(app, future))
    if request_log is not None:
        request_log.emit("request_accepted")
    # Cancellation of the consumer must not cancel queued/running producer work.
    result = asyncio.shield(producer)
    # Also observe an abandoned result, including callers that add another shield.
    result.add_done_callback(lambda future: _observe_completion(app, future))
    return result


def _run_inference(
    runtime: Runtime,
    request: ChatCompletionRequest,
    guard: LockType,
    loop: asyncio.AbstractEventLoop,
    streamer: TextIteratorStreamer | None,
    prepared: asyncio.Future[None] | None,
    request_log: _RequestLog | None,
) -> GenerationResult:
    started = perf_counter()
    result = None
    outcome = "completed"
    error_code = None
    exception_type = None

    def on_prepared():
        loop.call_soon_threadsafe(_resolve_preparation, prepared, None)

    try:
        result = generate(
            runtime,
            [message.model_dump() for message in request.messages],
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            streamer=streamer,
            on_prepared=on_prepared if prepared is not None else None,
        )
        return result
    except BaseException as error:
        if isinstance(error, ContextLimitError):
            outcome, error_code = "rejected", "context_length_exceeded"
        else:
            outcome, error_code = "failed", "generation_failed"
            exception_type = type(error).__name__
        if prepared is not None:
            loop.call_soon_threadsafe(_resolve_preparation, prepared, error)
        raise
    finally:
        duration_ms = (perf_counter() - started) * 1000
        guard.release()
        if request_log is not None:
            loop.call_soon_threadsafe(
                request_log.producer_finished, outcome, duration_ms,
                result.prompt_tokens if result is not None else None,
                len(result.generated_ids) if result is not None else None,
                error_code, exception_type,
            )


def _resolve_preparation(future: asyncio.Future[None], error: BaseException | None):
    # Runs on the event loop; failure may follow an already-signaled preparation.
    if not future.done():
        if error is None:
            future.set_result(None)
        else:
            future.set_exception(error)


def _observe_completion(app: FastAPI, future: asyncio.Future) -> None:
    app.state.producer_futures.discard(future)
    if not future.cancelled():
        # Observing does not consume the exception for other awaiters.
        future.exception()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = app.state
    state.initialized = False
    state.shutting_down = False
    state.runtime = None
    state.executor = None
    state.admission_guard = Lock()
    state.producer_futures = set()

    try:
        # Startup must finish before requests can use the model.
        state.runtime = load_runtime()
        state.executor = ThreadPoolExecutor(max_workers=1)
        state.initialized = True
        yield
    finally:
        state.initialized = False
        state.shutting_down = True
        if state.executor is not None:
            # Retain the runtime until all worker computation has actually ended.
            await asyncio.to_thread(state.executor.shutdown, wait=True)
            state.executor = None
        if state.producer_futures:
            await asyncio.gather(*state.producer_futures, return_exceptions=True)
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


async def _is_disconnected(request: Request) -> bool:
    disconnected = await request.is_disconnected()
    # Starlette's AnyIO probe can consume a concurrent Task.cancel(). Preserve it.
    if asyncio.current_task().cancelling():
        raise asyncio.CancelledError()
    return disconnected


async def _await_connected(request: Request, future: asyncio.Future):
    try:
        while not future.done():
            if await _is_disconnected(request):
                raise ClientDisconnect()
            await asyncio.wait({future}, timeout=0.1)
        if await _is_disconnected(request):
            raise ClientDisconnect()
        return future.result()
    except (ClientDisconnect, asyncio.CancelledError):
        # All result futures passed here are shielded; producer ownership stays intact.
        future.cancel()
        raise


def _next_text(streamer: TextIteratorStreamer) -> tuple[str, str]:
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
    streamer: TextIteratorStreamer,
    future: asyncio.Future[GenerationResult],
    request_id: str,
    created: int,
):
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
            outcome, text = await asyncio.to_thread(_next_text, streamer)
            if await _is_disconnected(request):
                return
            if outcome == "text":
                if text:
                    yield chunk({"content": text})
            elif outcome == "end":
                result = await _await_connected(request, future)
                break
            elif producer_done:
                # Work was complete before the empty queue read, so no text is lost.
                result = future.result()
                break
        yield chunk({}, result.finish_reason, _usage(result))
        yield "data: [DONE]\n\n"
    except ClientDisconnect:
        return
    except Exception as error:
        request.state.request_log.error("generation_failed", error)
        if await _is_disconnected(request):
            return
        yield _sse(_error_payload(
            "generation_failed", "The completion could not be generated.", request_id
        ))
        yield "data: [DONE]\n\n"
    finally:
        # This is the shielded delivery future, never the retained producer.
        future.cancel()


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    request_log = request.state.request_log
    request_log.stream = body.stream
    request_id = request_log.request_id
    created = int(time())
    try:
        if body.stream:
            state = request.app.state
            if not getattr(state, "initialized", False) or state.shutting_down:
                raise InferenceUnavailable("Inference is not available.")
            streamer = TextIteratorStreamer(
                state.runtime.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                timeout=0.1,
            )
            prepared = asyncio.get_running_loop().create_future()
            future = submit_inference(
                request.app, body, streamer=streamer, prepared=prepared,
                request_log=request_log,
            )
            await _await_connected(request, asyncio.shield(prepared))
            if future.done():
                future.result()  # A failure already known before headers is still HTTP.
            return StreamingResponse(
                _stream_completion(request, streamer, future, request_id, created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        result = await _await_connected(
            request, submit_inference(request.app, body, request_log=request_log)
        )
    except ClientDisconnect:
        return _DisconnectedResponse()
    except InferenceUnavailable:
        return _error_response(503, "not_ready", "Inference is not available.", request_log)
    except InferenceBusy:
        return _error_response(
            429, "server_busy", "Another inference request is running.", request_log
        )
    except ContextLimitError as error:
        # This exception is constructed from token counts, never prompt content.
        return _error_response(400, "context_length_exceeded", str(error), request_log)
    except Exception as error:
        return _error_response(
            500, "generation_failed", "The completion could not be generated.",
            request_log, error,
        )

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
    if getattr(state, "shutting_down", False):
        status = "shutting_down"
    elif not getattr(state, "initialized", False):
        status = "not_ready"
    elif state.admission_guard.locked():
        status = "busy"
    else:
        status = "ready"
    available = status == "ready"
    return JSONResponse(
        status_code=200 if available else 503,
        content={"ready": available, "status": status},
    )
