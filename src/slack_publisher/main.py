"""NATS JetStream worker that posts human-review cards to Slack.

Uses a push subscription with a queue group so the server distributes each
message to one of several running instances.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket

from pydantic import ValidationError

from agent_team.settings import get_settings
from review_events.jetstream import decode_event, run_queue_worker
from review_events.models import HumanReviewRequested
from slack_publisher.slack import SlackDeliveryError, post_review_to_slack
from telemetry.otel import configure_otel, start_span

logger = logging.getLogger(__name__)


def _instance_name() -> str:
    return f"slack-publisher@{socket.gethostname()}:{os.getpid()}"


async def consume_forever() -> None:
    settings = get_settings()
    configure_otel(default_service_name="slack-publisher", settings=settings)

    async def _on_message(msg) -> None:
        try:
            event = decode_event(msg, HumanReviewRequested)
        except ValidationError:
            logger.exception("Dropping invalid review event")
            await msg.ack()
            return

        try:
            with start_span(
                "slack.post_review",
                attributes={
                    "thread.id": event.thread_id,
                    "messaging.message.id": event.event_id,
                },
            ):
                post_review_to_slack(event, settings)
        except SlackDeliveryError:
            logger.exception("Failed to deliver event %s; will retry", event.event_id)
            await msg.nak()
            return

        await msg.ack()
        logger.info("Acked review event %s", event.event_id)

    await run_queue_worker(
        nats_url=settings.nats_url,
        client_name=_instance_name(),
        stream=settings.nats_stream,
        subject=settings.nats_subject,
        queue=settings.nats_queue,
        handler=_on_message,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(consume_forever())


if __name__ == "__main__":
    main()
