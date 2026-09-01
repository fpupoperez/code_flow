"""Apply Slack review feedback to a paused LangGraph thread."""

from __future__ import annotations

import logging

from langgraph_sdk import get_client

from agent_team.settings import Settings
from review_events.models import HumanReviewSubmitted

logger = logging.getLogger(__name__)


class WorkflowResumeError(RuntimeError):
    pass


async def resume_workflow(event: HumanReviewSubmitted, settings: Settings) -> None:
    """Inject feedback into the paused checkpoint, then continue the run.

    The graph is interrupted *before* ``human_review``. Updating state without
    ``as_node="human_review"`` keeps that node as the next step so it can read
    the new feedback. Using ``as_node="human_review"`` would skip the node and
    leave ``next_action`` stuck on ``human_review``.
    """
    client = get_client(url=settings.langgraph_server_url)
    logger.info(
        "Resuming thread %s (%s) from event %s",
        event.thread_id,
        event.action,
        event.event_id,
    )
    try:
        await client.threads.update_state(
            thread_id=event.thread_id,
            values={"feedback": event.feedback},
        )
        await client.runs.create(
            thread_id=event.thread_id,
            assistant_id=event.assistant_id or settings.langgraph_assistant_id,
        )
    except Exception as exc:
        raise WorkflowResumeError(str(exc)) from exc
