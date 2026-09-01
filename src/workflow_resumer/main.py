"""NATS JetStream worker that resumes LangGraph runs from Slack feedback.

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
from review_events.models import HumanReviewSubmitted
from telemetry.otel import configure_otel, start_span
from workflow_resumer.resume import WorkflowResumeError, resume_workflow

logger = logging.getLogger(__name__)


def _instance_name() -> str:
    return f"workflow-resumer@{socket.gethostname()}:{os.getpid()}"


async def consume_forever() -> None:
    settings = get_settings()
    configure_otel(default_service_name="workflow-resumer", settings=settings)

    async def _on_message(msg) -> None:
        try:
            event = decode_event(msg, HumanReviewSubmitted)
        except ValidationError:
            logger.exception("Dropping invalid review-feedback event")
            await msg.ack()
            return

        try:
            with start_span(
                "langgraph.resume",
                attributes={
                    "thread.id": event.thread_id,
                    "review.action": event.action,
                    "messaging.message.id": event.event_id,
                },
            ):
                await resume_workflow(event, settings)
        except WorkflowResumeError:
            logger.exception("Failed to resume thread %s; will retry", event.thread_id)
            await msg.nak()
            return

        await msg.ack()
        logger.info("Acked feedback event %s for thread %s", event.event_id, event.thread_id)

    await run_queue_worker(
        nats_url=settings.nats_url,
        client_name=_instance_name(),
        stream=settings.nats_stream,
        subject=settings.nats_feedback_subject,
        queue=settings.nats_feedback_queue,
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
