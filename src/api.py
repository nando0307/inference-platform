import asyncio
from _thread import LockType
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock
from time import time
from typing import Literal, Self
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError
from starlette.exceptions import HTTPException as StarletteHTTPException

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
        if self.stream:
            raise PydanticCustomError(
                "unsupported_parameter",
                "Streaming is not supported yet.",
            )
        return self


class InferenceUnavailable(RuntimeError):
    """The application is not accepting inference work."""


class InferenceBusy(RuntimeError):
    """Another producer owns the inference slot."""


def submit_inference(
    app: FastAPI, request: ChatCompletionRequest
) -> asyncio.Future[GenerationResult]:
    """Admit a validated request on the event loop and return a protected result."""
    loop = asyncio.get_running_loop()
    state = app.state
    if not getattr(state, "initialized", False) or state.shutting_down:
        raise InferenceUnavailable("Inference is not available.")
    if not state.admission_guard.acquire(blocking=False):
        raise InferenceBusy("Another inference request is running.")

    try:
        producer = loop.run_in_executor(
            state.executor, _run_inference, state.runtime, request, state.admission_guard
        )
    except Exception:
        state.admission_guard.release()
        raise

    state.producer_futures.add(producer)
    producer.add_done_callback(lambda future: _observe_completion(app, future))
    # Cancellation of the consumer must not cancel queued/running producer work.
    result = asyncio.shield(producer)
    # Also observe an abandoned result, including callers that add another shield.
    result.add_done_callback(lambda future: _observe_completion(app, future))
    return result


def _run_inference(
    runtime: Runtime, request: ChatCompletionRequest, guard: LockType
) -> GenerationResult:
    try:
        return generate(
            runtime,
            [message.model_dump() for message in request.messages],
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
    finally:
        guard.release()


def _observe_completion(app: FastAPI, future: asyncio.Future[GenerationResult]) -> None:
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


def _error_response(
    status_code: int, code: str, message: str, request_id: str
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {"code": code, "message": message, "request_id": request_id}
        },
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
    elif error_type == "unsupported_parameter":
        code = error_type
        message = "Streaming is not supported yet."
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


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    request_id = uuid4().hex
    created = int(time())
    try:
        result = await submit_inference(request.app, body)
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

    completion_tokens = len(result.generated_ids)
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
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": result.prompt_tokens + completion_tokens,
            },
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
