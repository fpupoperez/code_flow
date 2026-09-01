"""OpenTelemetry SDK bootstrap from environment variables.

Langfuse / MLflow / LangSmith trace the LangGraph LLM path. This module traces
the *other* processes: the FastAPI Slack webhook and the NATS workers. Those
processes can run on a laptop, in Compose, or on a remote host — so every
collector URL, protocol, and token comes from ``OTEL_*`` (or the matching
settings fields). Nothing is hardcoded to ``localhost`` or a Docker DNS name.

Standard variables (also accepted by the official OTEL exporters):

* ``OTEL_SDK_DISABLED`` — set ``true`` to skip the SDK entirely
* ``OTEL_SERVICE_NAME`` — process identity in the backend
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` — collector or vendor OTLP base URL
* ``OTEL_EXPORTER_OTLP_PROTOCOL`` — ``http/protobuf`` or ``grpc``
* ``OTEL_EXPORTER_OTLP_HEADERS`` — auth, e.g. ``authorization=Bearer TOKEN``
* ``OTEL_RESOURCE_ATTRIBUTES`` — extra resource tags (``key=value,key=value``)

Without ``OTEL_EXPORTER_OTLP_ENDPOINT`` the SDK stays a no-op so unit tests
and a local CLI do not need a collector.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from agent_team.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_configured = False
_TRACER_NAME = "agent_team.telemetry"


def _parse_resource_attributes(raw: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if key:
            attributes[key] = value.strip()
    return attributes


def _sync_otel_env(settings: Settings, *, default_service_name: str) -> str:
    """Copy settings into the standard OTEL env vars the exporters read."""
    service_name = (
        os.getenv("OTEL_SERVICE_NAME")
        or settings.otel_service_name
        or default_service_name
    )
    os.environ["OTEL_SERVICE_NAME"] = service_name

    endpoint = settings.otel_exporter_otlp_endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT", ""
    )
    if endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint

    protocol = settings.otel_exporter_otlp_protocol or os.getenv(
        "OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"
    )
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = protocol

    headers = settings.otel_exporter_otlp_headers or os.getenv(
        "OTEL_EXPORTER_OTLP_HEADERS", ""
    )
    if headers:
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = headers

    resources = settings.otel_resource_attributes or os.getenv(
        "OTEL_RESOURCE_ATTRIBUTES", ""
    )
    if resources:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = resources

    return service_name


def otel_enabled(settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    if cfg.otel_sdk_disabled:
        return False
    return bool((cfg.otel_exporter_otlp_endpoint or "").strip())


def configure_otel(
    *,
    default_service_name: str,
    settings: Settings | None = None,
) -> bool:
    """Install a process-wide tracer provider. Idempotent. Returns True if live."""
    global _configured
    if _configured:
        return otel_enabled(settings)

    cfg = settings or get_settings()
    if not otel_enabled(cfg):
        logger.debug("OpenTelemetry disabled (no OTEL_EXPORTER_OTLP_ENDPOINT)")
        _configured = True
        return False

    service_name = _sync_otel_env(cfg, default_service_name=default_service_name)
    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    resource_attrs = {
        "service.name": service_name,
        **_parse_resource_attributes(os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")),
    }

    try:
        if protocol.startswith("grpc"):
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

        exporter = OTLPSpanExporter()
        provider = TracerProvider(resource=Resource.create(resource_attrs))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        logger.warning("OpenTelemetry setup failed: %s", exc)
        _configured = True
        return False

    logger.info(
        "OpenTelemetry enabled service=%s endpoint=%s protocol=%s",
        service_name,
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        protocol,
    )
    _configured = True
    return True


def instrument_fastapi(app: Any) -> None:
    """Attach ASGI middleware. No-op when the SDK was not configured."""
    if not otel_enabled():
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health,/health")
    except Exception as exc:
        logger.warning("FastAPI OpenTelemetry instrumentation skipped: %s", exc)


@contextmanager
def start_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Create a span. Harmless when the tracer provider is the no-op default."""
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name, kind=kind) as span:
        for key, value in (attributes or {}).items():
            if value is None:
                continue
            span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise


def reset_otel_for_tests() -> None:
    global _configured
    _configured = False
