"""Build Slack Block Kit cards from human-review events."""

from __future__ import annotations

from review_events.models import HumanReviewRequested

_MAX_CODE_CHARS = 800
_MAX_TEXT_CHARS = 1500


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n..."


def build_review_blocks(event: HumanReviewRequested) -> list[dict]:
    code = _truncate(event.current_code or "No code generated yet.", _MAX_CODE_CHARS)
    notes = _truncate(event.editor_notes or "Ready for review.", _MAX_TEXT_CHARS)
    request = _truncate(event.original_request or "(none)", _MAX_TEXT_CHARS)
    research = _truncate(event.research_notes or "(none)", _MAX_TEXT_CHARS)
    thread_id = event.thread_id
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":rotating_light: *Human Review Required*\n"
                    f"Thread: `{thread_id}` · Event: `{event.event_id}`"
                ),
            },
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Original request:*\n{request}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Research notes:*\n{research}"},
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Editor notes:*\n_{notes}_"},
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Generated artifact:*\n```python\n{code}\n```",
            },
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"Assistant `{event.assistant_id}` · step {event.step_count} · "
                        f"prior feedback: {event.feedback or 'none'}"
                    ),
                }
            ],
        },
        {
            "type": "input",
            "block_id": "feedback_block",
            "optional": True,
            "element": {
                "type": "plain_text_input",
                "action_id": "feedback_input",
                "placeholder": {
                    "type": "plain_text",
                    "text": "If rejecting, type required updates here...",
                },
                "multiline": True,
            },
            "label": {"type": "plain_text", "text": "Revision Feedback Comments"},
        },
        {
            "type": "actions",
            "block_id": "review_actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Approve & Finalize"},
                    "style": "primary",
                    "action_id": "approve_btn",
                    "value": thread_id,
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Reject / Request Fix"},
                    "style": "danger",
                    "action_id": "reject_btn",
                    "value": thread_id,
                },
            ],
        },
    ]
