from unittest.mock import AsyncMock, Mock

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from agent_team.nodes.review_publisher import build_review_event, review_publisher_node
from agent_team.settings import get_settings
from review_events.jetstream import covering_subjects, push_subscribe
from review_events.models import HumanReviewRequested
from slack_publisher.blocks import build_review_blocks


def test_build_review_event_includes_required_fields() -> None:
    state = {
        "messages": [
            HumanMessage(content="Write a palindrome checker."),
            AIMessage(content="Looks ready."),
        ],
        "research_notes": "Need a case-insensitive function.",
        "current_code": "def is_palindrome(text: str) -> bool:\n    return text == text[::-1]\n",
        "editor_notes": "Add normalization of whitespace.",
        "feedback": "",
        "step_count": 4,
    }
    config = {"configurable": {"thread_id": "thread-99"}}

    event = build_review_event(state, config)

    assert event.event_type == "human_review_required"
    assert event.thread_id == "thread-99"
    assert event.original_request == "Write a palindrome checker."
    assert "case-insensitive" in event.research_notes
    assert "is_palindrome" in event.current_code
    assert "whitespace" in event.editor_notes
    assert event.step_count == 4
    assert len(event.recent_messages) == 2
    roundtrip = HumanReviewRequested.model_validate_json(event.model_dump_json())
    assert roundtrip.thread_id == event.thread_id


def test_slack_blocks_embed_thread_id_on_buttons() -> None:
    event = HumanReviewRequested(
        thread_id="thread-42",
        original_request="Compute fibonacci",
        current_code="def fib(n): ...",
        editor_notes="Looks good",
    )
    blocks = build_review_blocks(event)
    actions = next(block for block in blocks if block["type"] == "actions")
    values = {el["action_id"]: el["value"] for el in actions["elements"]}
    assert values["approve_btn"] == "thread-42"
    assert values["reject_btn"] == "thread-42"
    serialized = str(blocks)
    assert "Compute fibonacci" in serialized
    assert "Looks good" in serialized


@pytest.mark.asyncio
async def test_review_publisher_skips_when_nats_disabled(monkeypatch) -> None:
    monkeypatch.setenv("NATS_ENABLED", "false")
    get_settings.cache_clear()
    try:
        result = await review_publisher_node(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}},
        )
        assert result == {}
    finally:
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_push_subscribe_binds_queue_group() -> None:
    js = AsyncMock()
    stream_info = Mock()
    stream_info.config.subjects = ["agent.review.>"]
    js.stream_info = AsyncMock(return_value=stream_info)
    js.subscribe = AsyncMock(return_value=Mock())
    callback = AsyncMock()

    await push_subscribe(
        js,
        stream="AGENT_REVIEW",
        subject="agent.review.required",
        queue="slack-publisher",
        cb=callback,
    )

    js.subscribe.assert_awaited_once()
    args, kwargs = js.subscribe.await_args
    assert args[0] == "agent.review.required"
    assert kwargs["queue"] == "slack-publisher"
    assert kwargs["durable"] == "slack-publisher"
    assert kwargs["cb"] is callback
    assert kwargs["manual_ack"] is True
    assert kwargs["config"].deliver_group == "slack-publisher"
    assert kwargs["config"].durable_name == "slack-publisher"


def test_covering_subjects_share_one_stream() -> None:
    assert covering_subjects("agent.review.required") == ["agent.review.>"]
    assert covering_subjects("agent.review.feedback") == ["agent.review.>"]


def test_build_review_event_falls_back_to_last_message() -> None:
    event = build_review_event(
        {
            "messages": [
                HumanMessage(content="Write a palindrome checker."),
                AIMessage(content="Ready for review."),
            ],
            "current_code": "def is_palindrome(): ...",
        },
        {"configurable": {"thread_id": "thread-fallback"}},
    )
    assert event.editor_notes == "Ready for review."
    assert event.last_message == "Ready for review."
    assert event.original_request == "Write a palindrome checker."
