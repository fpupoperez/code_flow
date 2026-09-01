from unittest.mock import Mock

import pytest

from agent_team.settings import Settings
from telemetry.otel import (
    _parse_resource_attributes,
    configure_otel,
    instrument_fastapi,
    otel_enabled,
    reset_otel_for_tests,
    start_span,
)


def _settings(**kwargs) -> Settings:
    return Settings.model_construct(**kwargs)


def test_otel_disabled_without_endpoint() -> None:
    assert otel_enabled(_settings(otel_sdk_disabled=False, otel_exporter_otlp_endpoint="")) is False


def test_otel_disabled_by_flag_even_with_endpoint() -> None:
    assert (
        otel_enabled(
            _settings(
                otel_sdk_disabled=True,
                otel_exporter_otlp_endpoint="http://otel.example.com:4318",
            )
        )
        is False
    )


def test_otel_enabled_when_endpoint_is_set() -> None:
    assert (
        otel_enabled(
            _settings(
                otel_sdk_disabled=False,
                otel_exporter_otlp_endpoint="https://otlp.vendor.example/v1/traces",
            )
        )
        is True
    )


def test_start_span_is_safe_without_a_collector() -> None:
    with start_span("nats.publish", attributes={"messaging.system": "nats"}) as span:
        assert span is not None


def test_start_span_reraises_and_records() -> None:
    with pytest.raises(ValueError, match="boom"):
        with start_span("nats.consume"):
            raise ValueError("boom")


def test_parse_resource_attributes() -> None:
    assert _parse_resource_attributes("service.ns=prod, env=dev") == {
        "service.ns": "prod",
        "env": "dev",
    }
    assert _parse_resource_attributes("nokeep") == {}
    assert _parse_resource_attributes("") == {}


def test_configure_otel_is_noop_without_endpoint() -> None:
    reset_otel_for_tests()
    try:
        enabled = configure_otel(
            default_service_name="unit-test",
            settings=_settings(otel_exporter_otlp_endpoint="", otel_sdk_disabled=False),
        )
        assert enabled is False
    finally:
        reset_otel_for_tests()


def test_instrument_fastapi_is_safe_when_disabled() -> None:
    instrument_fastapi(Mock())
