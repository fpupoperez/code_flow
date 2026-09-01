import json
from urllib.parse import parse_qs
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from agent_team.settings import Settings
from review_events.models import HumanReviewRequested
from slack_emulator.app import create_app, post_approve_to_gateway
from slack_emulator.payloads import build_approve_payload, thread_id_from_blocks
from slack_gateway.payloads import parse_review_payload
from slack_gateway.signatures import verify_slack_signature
from slack_publisher.blocks import build_review_blocks


def _event() -> HumanReviewRequested:
    return HumanReviewRequested(
        thread_id="thread-emu-1",
        original_request="Write fibonacci",
        current_code="def fib(n): ...",
        editor_notes="Looks good",
    )


def test_emulator_health() -> None:
    settings = Settings.model_construct(slack_emulator_auto_approve=False)
    client = TestClient(create_app(settings))
    assert client.get("/health").json() == {"status": "ok"}


def test_emulator_rejects_card_without_thread() -> None:
    settings = Settings.model_construct(slack_emulator_auto_approve=False)
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/chat.postMessage",
        json={"channel": "C_EMULATOR", "text": "review", "blocks": []},
    )
    assert response.status_code == 400


def test_thread_id_from_blocks_raises_without_button() -> None:
    with pytest.raises(ValueError, match="approve_btn"):
        thread_id_from_blocks([])


def test_thread_id_from_approve_button() -> None:
    blocks = build_review_blocks(_event())
    assert thread_id_from_blocks(blocks) == "thread-emu-1"


def test_approve_payload_is_accepted_by_the_gateway_parser() -> None:
    payload = build_approve_payload(
        thread_id="thread-emu-1",
        channel="C_EMULATOR",
        text="review",
        blocks=build_review_blocks(_event()),
        message_ts="1.0",
    )
    decision = parse_review_payload(payload)
    assert decision.action == "approve"
    assert decision.feedback == "APPROVE"
    assert decision.thread_id == "thread-emu-1"


def test_emulator_accepts_form_encoded_blocks() -> None:
    settings = Settings.model_construct(
        slack_signing_secret="secret",
        slack_emulator_auto_approve=False,
        slack_channel="#agent-approvals",
    )
    event = _event()
    client = TestClient(create_app(settings))
    response = client.post(
        "/api/chat.postMessage",
        data={
            "channel": "#agent-approvals",
            "text": f"review {event.thread_id}",
            "blocks": json.dumps(build_review_blocks(event)),
        },
    )
    assert response.status_code == 200
    assert client.get("/reviews").json()["count"] == 1


def test_emulator_accepts_chat_post_message() -> None:
    settings = Settings.model_construct(
        slack_signing_secret="secret",
        slack_interactivity_url="http://gateway.test/slack/interactive",
        slack_emulator_auto_approve=False,
        slack_channel="#agent-approvals",
    )
    app = create_app(settings)
    event = _event()
    client = TestClient(app)
    response = client.post(
        "/api/chat.postMessage",
        json={
            "channel": "#agent-approvals",
            "text": f"review {event.thread_id}",
            "blocks": build_review_blocks(event),
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["ts"]
    inbox = client.get("/reviews").json()
    assert inbox["count"] == 1
    assert inbox["objects"][0]["thread_id"] == "thread-emu-1"


@pytest.mark.asyncio
async def test_emulator_webhook_is_signed_like_slack() -> None:
    settings = Settings.model_construct(
        slack_signing_secret="secret",
        slack_interactivity_url="http://gateway.test/slack/interactive",
        slack_emulator_approve_delay_seconds=0,
    )
    captured: dict[str, bytes | str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content
        captured["timestamp"] = request.headers["X-Slack-Request-Timestamp"]
        captured["signature"] = request.headers["X-Slack-Signature"]
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    with patch("slack_emulator.app.httpx.AsyncClient", client_factory):
        await post_approve_to_gateway(
            settings=settings,
            thread_id="thread-emu-1",
            channel="C_EMULATOR",
            text="review",
            blocks=build_review_blocks(_event()),
            message_ts="1.0",
        )

    raw = captured["body"]
    assert isinstance(raw, (bytes, bytearray))
    assert verify_slack_signature(
        signing_secret="secret",
        body=raw,
        timestamp=str(captured["timestamp"]),
        signature=str(captured["signature"]),
    )
    payload = json.loads(parse_qs(raw.decode())["payload"][0])
    decision = parse_review_payload(payload)
    assert decision.action == "approve"
    assert decision.thread_id == "thread-emu-1"
    assert str(captured["url"]).endswith("/slack/interactive")
