"""Process telemetry for the Slack gateway and NATS workers."""

from telemetry.otel import configure_otel, instrument_fastapi, start_span

__all__ = ["configure_otel", "instrument_fastapi", "start_span"]
