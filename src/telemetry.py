"""Opt-in, metadata-only HTTP observations, independent of inference scheduling."""

import asyncio
import logging
import math
import os
from urllib.parse import urlsplit
from uuid import UUID

from .vllm_engine import MODEL_SPECS


logger = logging.getLogger("inference.telemetry")
FLUSH_TIMEOUT_SECONDS = 3.0
_provider = None
_configuration = None
_OUTCOMES = {"completed", "rejected", "failed", "disconnected"}
_ERROR_CODES = {
    "invalid_request", "invalid_parameter_combination", "request_too_large",
    "not_ready", "server_busy", "context_length_exceeded", "generation_failed",
    "slow_consumer", "request_canceled", "unauthorized", "engine_failed",
    "load_failed", "upstream_failed", "request_timeout", "client_disconnected",
    "cleanup_failed", "internal_error",
}


def create_tracer():
    """Import and initialize the SDK only after explicit, complete opt-in."""
    enabled = os.environ.get("LANGFUSE_TRACING_ENABLED", "false").lower()
    if enabled != "true":
        if enabled not in {"", "false"}:
            logger.warning("Langfuse disabled: invalid opt-in setting.")
        return None
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    base_url = os.environ.get("LANGFUSE_BASE_URL", "")
    try:
        if not all(
            value and all(33 <= ord(char) <= 126 for char in value)
            for value in (public_key, secret_key, base_url)
        ):
            raise ValueError
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError
    except ValueError:
        logger.warning("Langfuse disabled: missing or invalid configuration.")
        return None

    global _configuration
    configuration = (public_key, secret_key, base_url)
    if _configuration is not None and _configuration != configuration:
        logger.warning("Langfuse disabled: configuration changes require a process restart.")
        return None

    try:
        from langfuse import Langfuse
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        # Do not attach an exporter to the process-global provider: other libraries
        # may record prompts, outputs, headers, or unrelated application spans there.
        global _provider
        if _provider is None:
            _provider = TracerProvider(resource=Resource({"service.name": "inference-platform"}))
        client = Langfuse(
            public_key=public_key, secret_key=secret_key, base_url=base_url,
            tracer_provider=_provider, timeout=2,
        )
        _configuration = configuration
        return client
    except Exception:
        logger.warning("Langfuse disabled: initialization failed.")
        return None


async def close_tracer(client):
    """Bound the lifespan's off-loop flush; SDK atexit owns process shutdown."""
    if client is None or _provider is None:
        return
    # Langfuse caches resources by public key even after shutdown. Keep its shared
    # clients alive across restarted/overlapping app lifespans. Only spans need
    # flushing: this helper never creates scores, media, or prompt-cache work.
    try:
        flushed = await asyncio.wait_for(
            asyncio.to_thread(_provider.force_flush, timeout_millis=2000),
            timeout=FLUSH_TIMEOUT_SECONDS,
        )
        if not flushed:
            logger.warning("Langfuse flush did not complete before its deadline.")
    except Exception:
        logger.warning("Langfuse flush failed or timed out.")


def start_request(client, request_id: str, *, backend: str):
    """Start a root generation using only a server-generated request identifier."""
    if client is None:
        return None
    try:
        if backend not in {"cpu", "model-router"} or type(request_id) is not str:
            return None
        parsed_id = UUID(request_id)
        if parsed_id.hex != request_id or parsed_id.int == 0:
            return None
        from opentelemetry.context import Context, attach, detach

        # Explicit trace_context overrides parentage; an empty context also avoids
        # inheriting arbitrary metadata/baggage from unrelated instrumentation.
        token = attach(Context())
        try:
            observation = client.start_observation(
                name="chat.completion", as_type="generation",
                trace_context={"trace_id": request_id},
                metadata={"request_id": request_id, "backend": backend},
            )
        finally:
            detach(token)
        return RequestTrace(observation)
    except Exception:
        logger.warning("Langfuse request observation could not be started.")
        return None


class RequestTrace:
    """Event-loop-owned allowlisted metadata; never retains request/response data."""

    def __init__(self, observation):
        self._observation = observation
        self._usage = {}

    def update(
        self, *, model=None, stream=None, prompt_tokens=None, completion_tokens=None,
        queue_wait_ms=None, producer_duration_ms=None,
    ):
        if self._observation is None:
            return
        fields = {}
        metadata = {}
        if type(model) is str and model in MODEL_SPECS:
            fields["model"] = model
        if type(stream) is bool:
            metadata["stream"] = stream
        for key, value in (
            ("queue_wait_ms", queue_wait_ms),
            ("producer_duration_ms", producer_duration_ms),
        ):
            if (
                type(value) is int and value >= 0
                or type(value) is float and value >= 0 and math.isfinite(value)
            ):
                metadata[key] = value
        for key, value in (("input", prompt_tokens), ("output", completion_tokens)):
            if type(value) is int and value >= 0:
                self._usage[key] = value
        if self._usage:
            fields["usage_details"] = self._usage.copy()
        if metadata:
            fields["metadata"] = metadata
        if fields:
            try:
                self._observation.update(**fields)
            except Exception:
                logger.warning("Langfuse request metadata could not be updated.")

    def finish(self, *, outcome: str, http_status=None, error_code=None):
        observation = self._observation
        if observation is None:
            return
        self._observation = None
        outcome = outcome if type(outcome) is str and outcome in _OUTCOMES else "failed"
        metadata = {"outcome": outcome}
        if type(http_status) is int and 100 <= http_status <= 599:
            metadata["http_status"] = http_status
        if type(error_code) is str and error_code in _ERROR_CODES:
            metadata["error_code"] = error_code
        level = (
            "ERROR" if outcome == "failed" or metadata.get("http_status", 0) >= 500
            else "DEFAULT" if outcome == "completed"
            else "WARNING"
        )
        try:
            observation.update(
                metadata=metadata, level=level,
                status_message=metadata.get("error_code", outcome),
            )
        except Exception:
            logger.warning("Langfuse request outcome could not be updated.")
        try:
            observation.end()
        except Exception:
            logger.warning("Langfuse request observation could not be ended.")
