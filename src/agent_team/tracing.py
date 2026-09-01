"""Observability backend for LangGraph runs.

Langfuse is the default tracer. MLflow and LangSmith are opt-in alternatives.

Why Langfuse first:

* MIT-licensed and fully self-hostable, so traces stay on our infrastructure.
  LangSmith self-hosting is an Enterprise contract; the default is their SaaS.
* Built for LLM traces (sessions, scores, prompt versions), not classical ML runs.
* Usage-based pricing with no seat fees.

Why MLflow exists as an option:

* Apache-2.0 tracking server many ML teams already run (or Databricks hosts).
* Use it when agent traces must sit next to training runs, metrics, artifacts,
  and a model registry — not when you only need LLM-ops.
* Activate it with ``MLFLOW_TRACKING_URI`` and leave Langfuse keys unset
  (or set ``LANGFUSE_ENABLED=false``).

LangSmith remains the LangChain-native fallback when neither Langfuse nor
MLflow is configured. That keeps zero-config Smith traces and the existing
eval harness without making a proprietary SaaS the production default.

Selection:

1. ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` → Langfuse
2. else ``MLFLOW_TRACKING_URI`` → MLflow
3. else ``LANGSMITH_TRACING`` / ``LANGCHAIN_TRACING_V2`` + API key → LangSmith
4. else no tracing

Langfuse uses the official LangChain ``CallbackHandler``. MLflow uses
``mlflow.langchain.autolog()``, which is the documented LangGraph path.
The Langfuse SDK already installs its OpenTelemetry exporter; we do not
register a second global ``TracerProvider``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from agent_team.settings import Settings, get_settings

logger = logging.getLogger(__name__)

TracingBackend = Literal["langfuse", "mlflow", "langsmith", "none"]

_configured_backend: TracingBackend | None = None
_langfuse_handler: Any = None


def resolve_tracing_backend(settings: Settings | None = None) -> TracingBackend:
    """Pick the tracer from settings. Langfuse wins when several are set.

    ``Settings`` already maps ``LANGFUSE_*``, ``MLFLOW_*``, ``LANGSMITH_*``,
    and the legacy ``LANGCHAIN_*`` aliases, so this function does not
    re-read ``os.environ``.
    """
    cfg = settings or get_settings()

    langfuse_keys = bool(cfg.langfuse_public_key and cfg.langfuse_secret_key)
    if cfg.langfuse_enabled and langfuse_keys:
        return "langfuse"

    if cfg.mlflow_enabled and cfg.mlflow_tracking_uri:
        return "mlflow"

    if cfg.langsmith_tracing and cfg.langsmith_api_key:
        return "langsmith"

    return "none"


def _sync_langfuse_env(settings: Settings) -> None:
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    if settings.langfuse_host:
        # Current SDK reads LANGFUSE_BASE_URL; older docs used LANGFUSE_HOST.
        os.environ["LANGFUSE_HOST"] = settings.langfuse_host
        os.environ["LANGFUSE_BASE_URL"] = settings.langfuse_host


def _sync_mlflow_env(settings: Settings) -> None:
    os.environ["MLFLOW_TRACKING_URI"] = settings.mlflow_tracking_uri
    if settings.mlflow_experiment_name:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = settings.mlflow_experiment_name
    if settings.mlflow_tracking_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    if settings.mlflow_tracking_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
    if settings.mlflow_tracking_token:
        os.environ["MLFLOW_TRACKING_TOKEN"] = settings.mlflow_tracking_token


def _disable_langsmith_env() -> None:
    """Avoid dual export when Langfuse or MLflow is the active backend."""
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


def _configure_langfuse(settings: Settings) -> None:
    _sync_langfuse_env(settings)
    _disable_langsmith_env()

    from langfuse import Langfuse, get_client

    # Instantiating the client registers the OTEL span processor that ships
    # traces to the self-hosted or cloud Langfuse OTLP endpoint.
    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host or None,
    )
    get_client()
    logger.info(
        "Tracing backend=langfuse host=%s (self-hostable MIT stack; other tracers disabled)",
        settings.langfuse_host or os.getenv("LANGFUSE_BASE_URL") or "sdk-default",
    )


def _configure_mlflow(settings: Settings) -> None:
    _sync_mlflow_env(settings)
    _disable_langsmith_env()

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    if settings.mlflow_experiment_name:
        mlflow.set_experiment(settings.mlflow_experiment_name)
    # Official LangGraph integration. run_tracer_inline keeps spans nested
    # when a node uses async invoke (the Slack-resume path is async).
    mlflow.langchain.autolog(log_traces=True, run_tracer_inline=True)
    logger.info(
        "Tracing backend=mlflow uri=%s experiment=%s "
        "(Langfuse keys were not set; using the experiment-platform alternative)",
        settings.mlflow_tracking_uri,
        settings.mlflow_experiment_name,
    )


def _configure_langsmith(settings: Settings) -> None:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    logger.info(
        "Tracing backend=langsmith project=%s "
        "(Langfuse and MLflow were not set; using the LangChain-native alternative)",
        settings.langsmith_project,
    )


def configure_tracing(settings: Settings | None = None) -> TracingBackend:
    """Idempotent process-start hook. Safe to call with no credentials."""
    global _configured_backend
    if _configured_backend is not None:
        return _configured_backend

    cfg = settings or get_settings()
    backend = resolve_tracing_backend(cfg)
    try:
        if backend == "langfuse":
            _configure_langfuse(cfg)
        elif backend == "mlflow":
            _configure_mlflow(cfg)
        elif backend == "langsmith":
            _configure_langsmith(cfg)
        else:
            logger.debug("Tracing backend=none (no Langfuse, MLflow, or LangSmith credentials)")
    except Exception as exc:
        logger.warning("Failed to configure tracing backend %s: %s", backend, exc)
        backend = "none"

    _configured_backend = backend
    return backend


def tracing_callbacks() -> list[Any]:
    """LangChain callbacks for LangGraph.

    LangSmith instruments via ``LANGSMITH_TRACING``. MLflow instruments via
    ``mlflow.langchain.autolog()`` at process start — attaching a second
    tracer here would duplicate spans.
    """
    global _langfuse_handler
    if configure_tracing() != "langfuse":
        return []
    if _langfuse_handler is None:
        from langfuse.langchain import CallbackHandler

        _langfuse_handler = CallbackHandler()
    return [_langfuse_handler]


def merge_invoke_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge tracing callbacks and session metadata into a LangGraph invoke config."""
    merged: dict[str, Any] = dict(config or {})
    callbacks = list(merged.get("callbacks") or [])
    callbacks.extend(tracing_callbacks())
    if callbacks:
        merged["callbacks"] = callbacks

    backend = configure_tracing()
    if backend in {"langfuse", "mlflow"}:
        merged.setdefault("run_name", "agent-team")
        metadata = dict(merged.get("metadata") or {})
        thread_id = (merged.get("configurable") or {}).get("thread_id")
        if thread_id:
            # Groups the HITL pause and the Slack-driven resume in one session.
            metadata.setdefault("thread_id", str(thread_id))
            if backend == "langfuse":
                metadata.setdefault("langfuse_session_id", str(thread_id))
        merged["metadata"] = metadata

    return merged


def attach_tracing(compiled: Any) -> Any:
    """Bind tracing callbacks on the compiled graph for LangGraph Server.

    The server invokes the exported ``graph`` object and never sees CLI
    ``merge_invoke_config``. ``with_config`` is the official Langfuse +
    LangGraph Server pattern; MLflow uses the same hook when a tracer exists.
    """
    callbacks = tracing_callbacks()
    if not callbacks:
        return compiled
    return compiled.with_config({"callbacks": callbacks, "run_name": "agent-team"})


def flush_tracing() -> None:
    """Flush pending Langfuse or MLflow spans. No-op for LangSmith and none."""
    backend = configure_tracing()
    if backend == "langfuse":
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception as exc:
            logger.debug("Langfuse flush skipped: %s", exc)
        return
    if backend == "mlflow":
        try:
            import mlflow

            flush = getattr(mlflow, "flush_trace", None) or getattr(
                mlflow, "flush_async_logging", None
            )
            if flush:
                flush()
        except Exception as exc:
            logger.debug("MLflow flush skipped: %s", exc)


def reset_tracing_for_tests() -> None:
    """Drop process-level tracing state. Tests only."""
    global _configured_backend, _langfuse_handler
    _configured_backend = None
    _langfuse_handler = None
