import importlib
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agent_team.settings import Settings
from review_events.models import HumanReviewSubmitted
from workflow_resumer.resume import WorkflowResumeError, resume_workflow

resumer_main = importlib.import_module("workflow_resumer.main")


def _event(**kwargs) -> HumanReviewSubmitted:
    defaults = {
        "thread_id": "thread-resume",
        "action": "approve",
        "feedback": "APPROVE",
        "assistant_id": "agent_team",
    }
    defaults.update(kwargs)
    return HumanReviewSubmitted(**defaults)


@pytest.mark.asyncio
async def test_resume_updates_state_without_as_node() -> None:
    client = AsyncMock()
    with patch("workflow_resumer.resume.get_client", return_value=client):
        await resume_workflow(
            _event(),
            Settings.model_construct(
                langgraph_server_url="http://langgraph:8123",
                langgraph_assistant_id="agent_team",
            ),
        )

    client.threads.update_state.assert_awaited_once()
    kwargs = client.threads.update_state.await_args.kwargs
    assert "as_node" not in kwargs
    assert kwargs["thread_id"] == "thread-resume"
    assert kwargs["values"] == {"feedback": "APPROVE"}
    client.runs.create.assert_awaited_once()
    create_kwargs = client.runs.create.await_args.kwargs
    assert create_kwargs["thread_id"] == "thread-resume"
    assert create_kwargs["assistant_id"] == "agent_team"


@pytest.mark.asyncio
async def test_resume_wraps_client_errors() -> None:
    client = AsyncMock()
    client.threads.update_state.side_effect = RuntimeError("server down")
    with patch("workflow_resumer.resume.get_client", return_value=client):
        with pytest.raises(WorkflowResumeError, match="server down"):
            await resume_workflow(
                _event(),
                Settings.model_construct(langgraph_server_url="http://langgraph:8123"),
            )


@pytest.mark.asyncio
async def test_resumer_acks_valid_event(monkeypatch) -> None:
    captured: dict = {}

    async def fake_worker(**kwargs):
        captured["handler"] = kwargs["handler"]
        captured["queue"] = kwargs["queue"]
        captured["subject"] = kwargs["subject"]

    monkeypatch.setattr(resumer_main, "run_queue_worker", fake_worker)
    monkeypatch.setattr(resumer_main, "configure_otel", lambda **_kwargs: False)
    monkeypatch.setattr(
        resumer_main,
        "get_settings",
        lambda: Settings.model_construct(
            nats_url="nats://x",
            nats_stream="AGENT_REVIEW",
            nats_feedback_subject="agent.review.feedback",
            nats_feedback_queue="workflow-resumer",
        ),
    )
    monkeypatch.setattr(resumer_main, "resume_workflow", AsyncMock())

    await resumer_main.consume_forever()
    assert captured["queue"] == "workflow-resumer"
    assert captured["subject"] == "agent.review.feedback"

    msg = Mock(data=_event().model_dump_json().encode())
    msg.ack = AsyncMock()
    msg.nak = AsyncMock()
    await captured["handler"](msg)
    msg.ack.assert_awaited_once()
    msg.nak.assert_not_awaited()


@pytest.mark.asyncio
async def test_resumer_acks_invalid_json_and_naks_resume_failure(monkeypatch) -> None:
    captured: dict = {}

    async def fake_worker(**kwargs):
        captured["handler"] = kwargs["handler"]

    monkeypatch.setattr(resumer_main, "run_queue_worker", fake_worker)
    monkeypatch.setattr(resumer_main, "configure_otel", lambda **_kwargs: False)
    monkeypatch.setattr(
        resumer_main,
        "get_settings",
        lambda: Settings.model_construct(
            nats_url="nats://x",
            nats_stream="S",
            nats_feedback_subject="s",
            nats_feedback_queue="q",
        ),
    )
    monkeypatch.setattr(
        resumer_main,
        "resume_workflow",
        AsyncMock(side_effect=WorkflowResumeError("boom")),
    )
    await resumer_main.consume_forever()

    bad = Mock(data=b"not-json")
    bad.ack = AsyncMock()
    bad.nak = AsyncMock()
    await captured["handler"](bad)
    bad.ack.assert_awaited_once()

    good = Mock(data=_event().model_dump_json().encode())
    good.ack = AsyncMock()
    good.nak = AsyncMock()
    await captured["handler"](good)
    good.nak.assert_awaited_once()
    good.ack.assert_not_awaited()
