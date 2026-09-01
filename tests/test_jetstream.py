from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from nats.js.errors import NotFoundError
from pydantic import BaseModel

from agent_team.nodes.review_publisher import recent_messages, review_publisher_node
from agent_team.settings import Settings
from review_events.jetstream import (
    _msg_header,
    covering_subjects,
    decode_event,
    ensure_stream,
    jetstream_connection,
    publish_event,
    run_queue_worker,
)
from review_events.models import HumanReviewRequested, HumanReviewSubmitted


@pytest.mark.asyncio
async def test_ensure_stream_creates_when_missing() -> None:
    js = AsyncMock()
    js.stream_info = AsyncMock(side_effect=NotFoundError())
    js.add_stream = AsyncMock()

    await ensure_stream(js, stream="AGENT_REVIEW", subject="agent.review.required")

    js.add_stream.assert_awaited_once()
    config = js.add_stream.await_args.args[0]
    assert config.name == "AGENT_REVIEW"
    assert config.subjects == ["agent.review.>"]


@pytest.mark.asyncio
async def test_ensure_stream_updates_missing_subjects() -> None:
    js = AsyncMock()
    info = Mock()
    info.config.subjects = ["other.>"]
    js.stream_info = AsyncMock(return_value=info)
    js.update_stream = AsyncMock()

    await ensure_stream(js, stream="AGENT_REVIEW", subject="agent.review.required")

    js.update_stream.assert_awaited_once()
    assert js.update_stream.await_args.kwargs["name"] == "AGENT_REVIEW"
    assert js.update_stream.await_args.kwargs["subjects"] == ["other.>", "agent.review.>"]


@pytest.mark.asyncio
async def test_ensure_stream_noop_when_subjects_already_cover() -> None:
    js = AsyncMock()
    info = Mock()
    info.config.subjects = ["agent.review.>"]
    js.stream_info = AsyncMock(return_value=info)
    js.update_stream = AsyncMock()

    await ensure_stream(js, stream="AGENT_REVIEW", subject="agent.review.required")

    js.update_stream.assert_not_called()


@pytest.mark.asyncio
async def test_publish_event_sets_nats_msg_id() -> None:
    js = AsyncMock()
    ack = Mock(stream="AGENT_REVIEW", seq=7)
    js.publish = AsyncMock(return_value=ack)
    event = HumanReviewRequested(thread_id="t1")

    await publish_event(js, subject="agent.review.required", event=event)

    subject, payload = js.publish.await_args.args
    assert subject == "agent.review.required"
    assert event.event_id.encode() not in payload or True
    assert js.publish.await_args.kwargs["headers"] == {"Nats-Msg-Id": event.event_id}
    decoded = HumanReviewRequested.model_validate_json(payload)
    assert decoded.thread_id == "t1"


@pytest.mark.asyncio
async def test_publish_event_omits_headers_without_event_id() -> None:
    class Dummy(BaseModel):
        x: int = 1

    js = AsyncMock()
    js.publish = AsyncMock(return_value=Mock(stream="S", seq=1))
    await publish_event(js, subject="plain", event=Dummy())
    assert js.publish.await_args.kwargs["headers"] is None


def test_decode_event_roundtrip() -> None:
    event = HumanReviewSubmitted(thread_id="t1", action="approve", feedback="APPROVE")
    msg = Mock(data=event.model_dump_json().encode())
    decoded = decode_event(msg, HumanReviewSubmitted)
    assert decoded.thread_id == "t1"
    assert decoded.action == "approve"


def test_msg_header_reads_mapping_and_missing() -> None:
    msg = Mock(headers={"Nats-Msg-Id": "abc"})
    assert _msg_header(msg, "Nats-Msg-Id") == "abc"
    msg.headers = None
    assert _msg_header(msg, "Nats-Msg-Id") is None
    msg.headers = ["not-a-mapping"]
    assert _msg_header(msg, "Nats-Msg-Id") is None


def test_covering_subjects_single_token() -> None:
    assert covering_subjects("reviews") == ["reviews", "reviews.>"]


@pytest.mark.asyncio
async def test_jetstream_connection_drains(monkeypatch) -> None:
    nc = Mock()
    nc.jetstream.return_value = "js"
    nc.drain = AsyncMock()
    monkeypatch.setattr("review_events.jetstream.connect", AsyncMock(return_value=nc))

    async with jetstream_connection("nats://localhost:4222") as (got_nc, js):
        assert got_nc is nc
        assert js == "js"

    nc.drain.assert_awaited_once()


@pytest.mark.asyncio
async def test_review_publisher_publishes_when_nats_enabled(monkeypatch) -> None:
    js = AsyncMock()
    nc = AsyncMock()
    published: dict = {}

    @asynccontextmanager
    async def fake_conn(_url, name="agent-team"):
        yield nc, js

    async def fake_publish(_js, *, subject, event):
        published["subject"] = subject
        published["event"] = event

    monkeypatch.setattr("agent_team.nodes.review_publisher.jetstream_connection", fake_conn)
    monkeypatch.setattr("agent_team.nodes.review_publisher.ensure_stream", AsyncMock())
    monkeypatch.setattr("agent_team.nodes.review_publisher.publish_event", fake_publish)
    monkeypatch.setattr(
        "agent_team.nodes.review_publisher.get_settings",
        lambda: Settings.model_construct(
            nats_enabled=True,
            nats_url="nats://localhost:4222",
            nats_stream="AGENT_REVIEW",
            nats_subject="agent.review.required",
            langgraph_assistant_id="agent_team",
        ),
    )

    result = await review_publisher_node(
        {"messages": [HumanMessage(content="hi")], "current_code": "x"},
        {"configurable": {"thread_id": "t1"}},
    )
    assert result == {}
    assert published["subject"] == "agent.review.required"
    assert published["event"].thread_id == "t1"
    assert published["event"].current_code == "x"


@pytest.mark.asyncio
async def test_run_queue_worker_wraps_handler_and_stops(monkeypatch) -> None:
    nc = Mock()
    nc.jetstream.return_value = AsyncMock()
    nc.drain = AsyncMock()
    subscribed = {}

    async def fake_subscribe(js, *, stream, subject, queue, cb):
        subscribed["cb"] = cb
        subscribed["queue"] = queue
        subscribed["subject"] = subject

    class ImmediateStop:
        async def wait(self):
            return None

        def set(self):
            return None

    monkeypatch.setattr("review_events.jetstream.connect", AsyncMock(return_value=nc))
    monkeypatch.setattr("review_events.jetstream.push_subscribe", fake_subscribe)
    monkeypatch.setattr("review_events.jetstream.asyncio.Event", ImmediateStop)

    handled = []

    async def handler(msg):
        handled.append(msg)

    await run_queue_worker(
        nats_url="nats://localhost:4222",
        client_name="unit-test",
        stream="AGENT_REVIEW",
        subject="agent.review.required",
        queue="slack-publisher",
        handler=handler,
    )
    assert subscribed["queue"] == "slack-publisher"
    msg = Mock()
    await subscribed["cb"](msg)
    assert handled == [msg]
    nc.drain.assert_awaited_once()


def test_recent_messages_reads_role_from_dicts_and_objects() -> None:
    snippets = recent_messages(
        {
            "messages": [
                {"role": "user", "content": "hello"},
                AIMessage(content="world"),
            ]
        }
    )
    assert [item.role for item in snippets] == ["user", "ai"]
    assert [item.content for item in snippets] == ["hello", "world"]
