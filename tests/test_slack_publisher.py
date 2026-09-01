import importlib
from unittest.mock import AsyncMock, Mock, patch

import pytest
from slack_sdk.errors import SlackApiError

from agent_team.settings import Settings
from review_events.models import HumanReviewRequested
from slack_publisher.blocks import build_review_blocks
from slack_publisher.slack import SlackDeliveryError, post_review_to_slack

publisher_main = importlib.import_module("slack_publisher.main")


def _event() -> HumanReviewRequested:
    return HumanReviewRequested(
        thread_id="thread-pub",
        original_request="Write fibonacci",
        current_code="def fib(n): ...",
        editor_notes="Looks good",
    )


def test_post_review_requires_enabled_token() -> None:
    with pytest.raises(SlackDeliveryError):
        post_review_to_slack(
            _event(), Settings.model_construct(slack_enabled=False, slack_bot_token="xoxb")
        )
    with pytest.raises(SlackDeliveryError):
        post_review_to_slack(
            _event(), Settings.model_construct(slack_enabled=True, slack_bot_token="")
        )


def test_post_review_normalizes_base_url_and_posts() -> None:
    with patch("slack_publisher.slack.WebClient") as web_client:
        web_client.return_value.chat_postMessage.return_value = {"ok": True}
        settings = Settings.model_construct(
            slack_enabled=True,
            slack_bot_token="xoxb-test",
            slack_channel="#agent-approvals",
            slack_api_base_url="http://emulator/api",
        )
        post_review_to_slack(_event(), settings)
        assert web_client.call_args.kwargs["base_url"] == "http://emulator/api/"
        web_client.return_value.chat_postMessage.assert_called_once()
        kwargs = web_client.return_value.chat_postMessage.call_args.kwargs
        assert kwargs["channel"] == "#agent-approvals"
        assert "thread-pub" in kwargs["text"]


def test_post_review_wraps_slack_api_error() -> None:
    with patch("slack_publisher.slack.WebClient") as web_client:
        web_client.return_value.chat_postMessage.side_effect = SlackApiError(
            "failed", {"error": "channel_not_found"}
        )
        with pytest.raises(SlackDeliveryError, match="channel_not_found"):
            post_review_to_slack(
                _event(),
                Settings.model_construct(
                    slack_enabled=True,
                    slack_bot_token="xoxb-test",
                    slack_channel="#agent-approvals",
                ),
            )


def test_blocks_truncate_long_code() -> None:
    event = HumanReviewRequested(thread_id="t", current_code="x" * 2000)
    blocks = build_review_blocks(event)
    artifact = next(block for block in blocks if "Generated artifact" in str(block))
    assert "\n..." in artifact["text"]["text"]


@pytest.mark.asyncio
async def test_publisher_acks_valid_event(monkeypatch) -> None:
    captured: dict = {}

    async def fake_worker(**kwargs):
        captured["handler"] = kwargs["handler"]
        captured["queue"] = kwargs["queue"]
        captured["subject"] = kwargs["subject"]

    posted: list = []
    monkeypatch.setattr(publisher_main, "run_queue_worker", fake_worker)
    monkeypatch.setattr(publisher_main, "configure_otel", lambda **_kwargs: False)
    monkeypatch.setattr(
        publisher_main,
        "get_settings",
        lambda: Settings.model_construct(
            nats_url="nats://x",
            nats_stream="AGENT_REVIEW",
            nats_subject="agent.review.required",
            nats_queue="slack-publisher",
        ),
    )
    monkeypatch.setattr(publisher_main, "post_review_to_slack", lambda event, _s: posted.append(event))

    await publisher_main.consume_forever()
    assert captured["queue"] == "slack-publisher"
    assert captured["subject"] == "agent.review.required"

    event = _event()
    msg = Mock(data=event.model_dump_json().encode())
    msg.ack = AsyncMock()
    msg.nak = AsyncMock()
    await captured["handler"](msg)
    msg.ack.assert_awaited_once()
    msg.nak.assert_not_awaited()
    assert posted[0].thread_id == "thread-pub"


@pytest.mark.asyncio
async def test_publisher_acks_invalid_json_and_naks_slack_failure(monkeypatch) -> None:
    captured: dict = {}

    async def fake_worker(**kwargs):
        captured["handler"] = kwargs["handler"]

    monkeypatch.setattr(publisher_main, "run_queue_worker", fake_worker)
    monkeypatch.setattr(publisher_main, "configure_otel", lambda **_kwargs: False)
    monkeypatch.setattr(
        publisher_main,
        "get_settings",
        lambda: Settings.model_construct(
            nats_url="nats://x",
            nats_stream="S",
            nats_subject="s",
            nats_queue="q",
        ),
    )

    def fail(_event, _settings):
        raise SlackDeliveryError("boom")

    monkeypatch.setattr(publisher_main, "post_review_to_slack", fail)
    await publisher_main.consume_forever()

    bad = Mock(data=b"not-json")
    bad.ack = AsyncMock()
    bad.nak = AsyncMock()
    await captured["handler"](bad)
    bad.ack.assert_awaited_once()
    bad.nak.assert_not_awaited()

    good = Mock(data=_event().model_dump_json().encode())
    good.ack = AsyncMock()
    good.nak = AsyncMock()
    await captured["handler"](good)
    good.nak.assert_awaited_once()
    good.ack.assert_not_awaited()
