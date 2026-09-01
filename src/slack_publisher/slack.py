"""Deliver human-review events from NATS JetStream to Slack."""

from __future__ import annotations

import logging

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from agent_team.settings import Settings
from review_events.models import HumanReviewRequested
from slack_publisher.blocks import build_review_blocks

logger = logging.getLogger(__name__)


class SlackDeliveryError(RuntimeError):
    pass


def post_review_to_slack(event: HumanReviewRequested, settings: Settings) -> None:
    if not settings.slack_enabled or not settings.slack_bot_token:
        raise SlackDeliveryError("Slack is disabled or SLACK_BOT_TOKEN is missing")

    base_url = (settings.slack_api_base_url or "https://slack.com/api/").rstrip("/") + "/"
    client = WebClient(token=settings.slack_bot_token, base_url=base_url)
    try:
        client.chat_postMessage(
            channel=settings.slack_channel,
            text=f"AI work pipeline interrupted: review thread `{event.thread_id}`",
            blocks=build_review_blocks(event),
        )
    except SlackApiError as exc:
        error = exc.response.get("error") if exc.response else str(exc)
        raise SlackDeliveryError(f"Slack API error: {error}") from exc

    logger.info("Posted Slack review card for thread %s", event.thread_id)
