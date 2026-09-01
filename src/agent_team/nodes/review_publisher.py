"""Publish a human-review event to NATS JetStream.

This node does not talk to Slack. A separate subscriber service consumes the
stream and delivers the Block Kit card.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from agent_team.settings import get_settings
from agent_team.state import AgentState
from agent_team.utils import message_text, original_request
from review_events.jetstream import ensure_stream, jetstream_connection, publish_event
from review_events.models import HumanReviewRequested, MessageSnippet

logger = logging.getLogger(__name__)

_MAX_SNIPPETS = 10
_MAX_SNIPPET_CHARS = 2000


def _thread_id(config: RunnableConfig) -> str:
    configurable = (config or {}).get("configurable") or {}
    return str(configurable.get("thread_id") or "unknown_thread")


def _role_of(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("role") or message.get("type") or "unknown")
    msg_type = getattr(message, "type", None)
    if msg_type:
        return str(msg_type)
    return message.__class__.__name__.lower()


def recent_messages(state: AgentState, limit: int = _MAX_SNIPPETS) -> list[MessageSnippet]:
    snippets: list[MessageSnippet] = []
    for message in list(state.get("messages") or [])[-limit:]:
        content = message_text(message)[:_MAX_SNIPPET_CHARS]
        snippets.append(MessageSnippet(role=_role_of(message), content=content))
    return snippets


def build_review_event(state: AgentState, config: RunnableConfig) -> HumanReviewRequested:
    settings = get_settings()
    editor_notes = state.get("editor_notes") or ""
    if not editor_notes and state.get("messages"):
        editor_notes = message_text(state["messages"][-1])
    last_message = ""
    if state.get("messages"):
        last_message = message_text(state["messages"][-1])
    return HumanReviewRequested(
        thread_id=_thread_id(config),
        assistant_id=settings.langgraph_assistant_id,
        original_request=original_request(state),
        research_notes=state.get("research_notes") or "",
        current_code=state.get("current_code") or "",
        editor_notes=editor_notes,
        feedback=state.get("feedback") or "",
        step_count=int(state.get("step_count") or 0),
        last_message=last_message,
        recent_messages=recent_messages(state),
    )


async def review_publisher_node(state: AgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    event = build_review_event(state, config)
    logger.info(
        "Publishing human-review event %s for thread %s",
        event.event_id,
        event.thread_id,
    )

    if not settings.nats_enabled:
        logger.warning("NATS is disabled; skipping review event publish")
        return {}

    async with jetstream_connection(settings.nats_url) as (_nc, js):
        await ensure_stream(js, stream=settings.nats_stream, subject=settings.nats_subject)
        await publish_event(js, subject=settings.nats_subject, event=event)

    return {}
