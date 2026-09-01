"""Build the interactivity payload Slack would POST after an Approve click."""

from __future__ import annotations

from typing import Any


def thread_id_from_blocks(blocks: list[Any]) -> str:
    """Read the review thread id from the Approve button ``value``."""
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "actions":
            continue
        for element in block.get("elements") or []:
            if not isinstance(element, dict):
                continue
            if element.get("action_id") == "approve_btn" and element.get("value"):
                return str(element["value"])
    raise ValueError("Review card has no approve_btn value (thread_id)")


def build_approve_payload(
    *,
    thread_id: str,
    channel: str,
    text: str,
    blocks: list[Any],
    message_ts: str,
) -> dict[str, Any]:
    """Minimal ``block_actions`` body the gateway already knows how to parse."""
    return {
        "type": "block_actions",
        "trigger_id": f"emulator-{message_ts}",
        "response_url": f"http://slack-emulator/hooks/{message_ts}",
        "user": {"id": "U_EMULATOR", "username": "slack-emulator", "name": "slack-emulator"},
        "channel": {"id": channel or "C_EMULATOR", "name": "agent-approvals"},
        "team": {"id": "T_EMULATOR"},
        "actions": [
            {
                "action_id": "approve_btn",
                "block_id": "review_actions",
                "value": thread_id,
                "type": "button",
            }
        ],
        "state": {"values": {}},
        "message": {
            "ts": message_ts,
            "text": text,
            "blocks": blocks,
        },
    }
