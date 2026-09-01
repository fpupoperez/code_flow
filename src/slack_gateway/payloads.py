"""Parse Slack interactive payloads and map them to graph feedback."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from review_events.models import HumanReviewSubmitted


class ReviewDecision(BaseModel):
    thread_id: str
    action: Literal["approve", "reject"]
    feedback: str


class IgnoredAction(BaseModel):
    action: Literal["ignored"] = "ignored"
    action_id: str = ""


def _first_action(payload: dict[str, Any]) -> dict[str, Any]:
    actions = payload.get("actions") or []
    if isinstance(actions, list) and actions:
        first = actions[0]
        if isinstance(first, dict):
            return first
    return {}


def extract_plain_text_feedback(payload: dict[str, Any], default: str) -> str:
    state_values = (payload.get("state") or {}).get("values") or {}
    for blocks in state_values.values():
        if not isinstance(blocks, dict):
            continue
        for action_obj in blocks.values():
            if isinstance(action_obj, dict) and action_obj.get("type") == "plain_text_input":
                value = action_obj.get("value")
                if value:
                    return str(value)
    return default


def parse_review_payload(payload: dict[str, Any]) -> ReviewDecision | IgnoredAction:
    action = _first_action(payload)
    action_id = str(action.get("action_id") or "")
    thread_id = str(action.get("value") or "")

    if action_id == "approve_btn":
        if not thread_id:
            raise ValueError("Missing thread_id on approve action")
        return ReviewDecision(thread_id=thread_id, action="approve", feedback="APPROVE")

    if action_id == "reject_btn":
        if not thread_id:
            raise ValueError("Missing thread_id on reject action")
        feedback = extract_plain_text_feedback(
            payload, default="Changes requested via Slack without comments."
        )
        return ReviewDecision(thread_id=thread_id, action="reject", feedback=feedback)

    return IgnoredAction(action_id=action_id)


def slack_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the Slack fields the workflow worker may need later."""
    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    channel = payload.get("channel") if isinstance(payload.get("channel"), dict) else {}
    return {
        "user": user,
        "channel": channel,
        "team": payload.get("team") if isinstance(payload.get("team"), dict) else {},
        "actions": payload.get("actions") or [],
        "state": payload.get("state") or {},
        "message_ts": ((payload.get("message") or {}) if isinstance(payload.get("message"), dict) else {}).get("ts"),
        "response_url": payload.get("response_url"),
        "trigger_id": payload.get("trigger_id"),
    }


def decision_from_slack(
    decision: ReviewDecision,
    payload: dict[str, Any],
    *,
    assistant_id: str,
) -> HumanReviewSubmitted:
    user = payload.get("user") if isinstance(payload.get("user"), dict) else {}
    channel = payload.get("channel") if isinstance(payload.get("channel"), dict) else {}
    action = _first_action(payload)
    return HumanReviewSubmitted(
        thread_id=decision.thread_id,
        assistant_id=assistant_id,
        action=decision.action,
        feedback=decision.feedback,
        slack_user_id=str(user.get("id") or ""),
        slack_username=str(user.get("username") or user.get("name") or ""),
        slack_channel=str(channel.get("id") or ""),
        slack_response_url=str(payload.get("response_url") or ""),
        slack_action_id=str(action.get("action_id") or ""),
        slack_trigger_id=str(payload.get("trigger_id") or ""),
        slack_context=slack_context(payload),
    )
