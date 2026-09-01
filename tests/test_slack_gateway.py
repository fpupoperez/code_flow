import json
from unittest.mock import AsyncMock, patch
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from contextlib import asynccontextmanager

from agent_team.settings import Settings
from review_events.models import HumanReviewSubmitted
from slack_gateway.app import create_app, publish_feedback_event
from slack_gateway.payloads import IgnoredAction, parse_review_payload
from slack_gateway.signatures import sign_slack_request, verify_slack_signature


def _settings(**kwargs) -> Settings:
    defaults = {
        "slack_signing_secret": "secret",
        "nats_enabled": True,
        "langgraph_assistant_id": "agent_team",
        "otel_exporter_otlp_endpoint": "",
        "otel_sdk_disabled": True,
    }
    defaults.update(kwargs)
    return Settings.model_construct(**defaults)


def _post_interactive(client: TestClient, payload: dict | None, *, secret: str = "secret"):
    raw = urlencode({"payload": json.dumps(payload)} if payload is not None else {}).encode()
    timestamp, signature = sign_slack_request(signing_secret=secret, body=raw)
    return client.post(
        "/slack/interactive",
        content=raw,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
    )


def test_health() -> None:
    client = TestClient(create_app(_settings()))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rejects_bad_signature() -> None:
    client = TestClient(create_app(_settings()))
    response = client.post(
        "/slack/interactive",
        content=b"payload=%7B%7D",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": "1000000000",
            "X-Slack-Signature": "v0=deadbeef",
        },
    )
    assert response.status_code == 401


def test_missing_payload_is_400() -> None:
    client = TestClient(create_app(_settings()))
    response = _post_interactive(client, None)
    assert response.status_code == 400


def test_ignored_action_is_acknowledged() -> None:
    client = TestClient(create_app(_settings()))
    response = _post_interactive(
        client, {"actions": [{"action_id": "other_btn", "value": "thread-1"}]}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ignored_action"


def test_missing_thread_id_is_400() -> None:
    client = TestClient(create_app(_settings()))
    response = _post_interactive(client, {"actions": [{"action_id": "approve_btn", "value": ""}]})
    assert response.status_code == 400


def test_nats_disabled_returns_503() -> None:
    client = TestClient(create_app(_settings(nats_enabled=False)))
    response = _post_interactive(
        client, {"actions": [{"action_id": "approve_btn", "value": "thread-1"}]}
    )
    assert response.status_code == 503


def test_approve_publishes_feedback_event() -> None:
    app = create_app(_settings())
    client = TestClient(app)
    with patch("slack_gateway.app.publish_feedback_event", new_callable=AsyncMock) as publish:
        response = _post_interactive(
            client,
            {
                "actions": [{"action_id": "approve_btn", "value": "thread-9"}],
                "user": {"id": "U1", "username": "ana"},
                "channel": {"id": "C1"},
            },
        )
    assert response.status_code == 200
    assert response.json()["response_type"] == "ephemeral"
    publish.assert_awaited_once()
    event = publish.await_args.args[0]
    assert event.action == "approve"
    assert event.feedback == "APPROVE"
    assert event.thread_id == "thread-9"
    assert event.slack_user_id == "U1"


def test_parse_ignored_and_reject_defaults() -> None:
    ignored = parse_review_payload({"actions": [{"action_id": "noop"}]})
    assert isinstance(ignored, IgnoredAction)
    decision = parse_review_payload(
        {"actions": [{"action_id": "reject_btn", "value": "t1"}], "state": {}}
    )
    assert decision.action == "reject"
    assert "without comments" in decision.feedback


def test_parse_actions_as_non_list_is_ignored() -> None:
    result = parse_review_payload({"actions": {"action_id": "approve_btn", "value": "t"}})
    assert isinstance(result, IgnoredAction)


def test_verify_rejects_stale_and_invalid_timestamps() -> None:
    body = b"payload=%7B%7D"
    timestamp, signature = sign_slack_request(signing_secret="secret", body=body, timestamp="1")
    assert not verify_slack_signature(
        signing_secret="secret",
        body=body,
        timestamp=timestamp,
        signature=signature,
        max_age_seconds=300,
    )
    assert not verify_slack_signature(
        signing_secret="secret",
        body=body,
        timestamp="not-int",
        signature=signature,
    )
    assert not verify_slack_signature(
        signing_secret="",
        body=body,
        timestamp="1",
        signature=signature,
    )


@pytest.mark.asyncio
async def test_publish_feedback_event_writes_to_jetstream(monkeypatch) -> None:
    published: dict = {}
    js = AsyncMock()

    @asynccontextmanager
    async def fake_conn(_url, name="slack-gateway"):
        yield AsyncMock(), js

    async def fake_publish(_js, *, subject, event):
        published["subject"] = subject
        published["event"] = event

    monkeypatch.setattr("slack_gateway.app.jetstream_connection", fake_conn)
    monkeypatch.setattr("slack_gateway.app.ensure_stream", AsyncMock())
    monkeypatch.setattr("slack_gateway.app.publish_event", fake_publish)

    event = HumanReviewSubmitted(thread_id="t1", action="approve", feedback="APPROVE")
    settings = _settings(
        nats_enabled=True,
        nats_url="nats://localhost:4222",
        nats_stream="AGENT_REVIEW",
        nats_feedback_subject="agent.review.feedback",
    )
    await publish_feedback_event(event, settings)
    assert published["subject"] == "agent.review.feedback"
    assert published["event"].thread_id == "t1"
