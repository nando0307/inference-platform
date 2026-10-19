import asyncio
import json
from _thread import LockType
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from queue import Empty
from threading import Lock
from time import time
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
from starlette.types import Receive, Scope, Send
from transformers import TextIteratorStreamer

from .inference import (
    MODEL_ID,
    ContextLimitError,
    GenerationResult,
    Runtime,
    generate,
    load_runtime,
)


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
            loop, streamer, prepared,
        )
    except Exception as error:
        state.admission_guard.release()
        if prepared is not None:
            _resolve_preparation(prepared, error)
        raise

    state.producer_futures.add(producer)
    producer.add_done_callback(lambda future: _observe_completion(app, future))
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
) -> GenerationResult:
    def on_prepared():
        loop.call_soon_threadsafe(_resolve_preparation, prepared, None)

    try:
        return generate(
            runtime,
            [message.model_dump() for message in request.messages],
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            streamer=streamer,
            on_prepared=on_prepared if prepared is not None else None,
        )
    except BaseException as error:
        if prepared is not None:
            loop.call_soon_threadsafe(_resolve_preparation, prepared, error)
        raise
    finally:
        guard.release()


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
    status_code: int, code: str, message: str, request_id: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(code, message, request_id),
        headers={"X-Request-ID": request_id},
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
            422, "invalid_request", "Request body must contain valid JSON.", uuid4().hex
        )
    return await http_exception_handler(request, error)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, error: RequestValidationError):
    request_id = uuid4().hex
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
    return _error_response(422, code, message, request_id)


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
    except Exception:
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
    request_id = uuid4().hex
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
                request.app, body, streamer=streamer, prepared=prepared
            )
            await _await_connected(request, asyncio.shield(prepared))
            if future.done():
                future.result()  # A failure already known before headers is still HTTP.
            return StreamingResponse(
                _stream_completion(request, streamer, future, request_id, created),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Request-ID": request_id},
            )
        result = await _await_connected(request, submit_inference(request.app, body))
    except ClientDisconnect:
        return _DisconnectedResponse()
    except InferenceUnavailable:
        return _error_response(503, "not_ready", "Inference is not available.", request_id)
    except InferenceBusy:
        return _error_response(
            429, "server_busy", "Another inference request is running.", request_id
        )
    except ContextLimitError as error:
        # This exception is constructed from token counts, never prompt content.
        return _error_response(400, "context_length_exceeded", str(error), request_id)
    except Exception:
        return _error_response(
            500, "generation_failed", "The completion could not be generated.", request_id
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
        headers={"X-Request-ID": request_id},
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
