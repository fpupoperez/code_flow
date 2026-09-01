"""Contract for human-review events published onto NATS JetStream."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class MessageSnippet(BaseModel):
    """A compact view of one conversation turn."""

    role: str
    content: str


class HumanReviewRequested(BaseModel):
    """Payload the graph publishes when an artifact is ready for human sign-off.

    The Slack microservice consumes this event and is the only component that
    talks to Slack. Keep this schema backward-compatible: add fields, do not
    rename or remove them.
    """

    event_type: Literal["human_review_required"] = "human_review_required"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    thread_id: str
    assistant_id: str = "agent_team"
    original_request: str = ""
    research_notes: str = ""
    current_code: str = ""
    editor_notes: str = ""
    feedback: str = ""
    step_count: int = 0
    last_message: str = ""
    recent_messages: list[MessageSnippet] = Field(default_factory=list)


class HumanReviewSubmitted(BaseModel):
    """Payload the Slack gateway publishes when a reviewer clicks Approve or Reject.

    ``workflow_resumer`` consumes this event and is the only component that
    talks to LangGraph Server. Keep this schema backward-compatible: add fields,
    do not rename or remove them.
    """

    event_type: Literal["human_review_submitted"] = "human_review_submitted"
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    published_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    thread_id: str
    assistant_id: str = "agent_team"
    action: Literal["approve", "reject"]
    feedback: str
    slack_user_id: str = ""
    slack_username: str = ""
    slack_channel: str = ""
    slack_response_url: str = ""
    slack_action_id: str = ""
    slack_trigger_id: str = ""
    slack_context: dict[str, Any] = Field(default_factory=dict)
