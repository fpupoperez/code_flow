from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

from agent_team.nodes.coder import coder_node
from agent_team.nodes.editor import editor_node
from agent_team.nodes.human_review import human_review_node
from agent_team.nodes.researcher import researcher_node
from agent_team.nodes.supervisor import Router, route_next, supervisor_node
from agent_team.settings import Settings


def test_route_next_defaults_to_researcher() -> None:
    assert route_next({}) == "researcher"
    assert route_next({"next_action": "coder"}) == "coder"


def test_supervisor_forces_review_at_step_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_team.nodes.supervisor.get_settings",
        lambda: Settings.model_construct(max_supervisor_steps=2),
    )
    result = supervisor_node({"messages": [], "step_count": 1}, {})
    assert result == {"next_action": "human_review", "step_count": 2}


def test_supervisor_uses_structured_choice(monkeypatch) -> None:
    class FakeStructured:
        def invoke(self, _msgs):
            return Router(next_step="coder", reason="research notes exist")

    class FakeLLM:
        def with_structured_output(self, *_args, **_kwargs):
            return FakeStructured()

    monkeypatch.setattr("agent_team.nodes.supervisor.get_llm", lambda: FakeLLM())
    monkeypatch.setattr(
        "agent_team.nodes.supervisor.get_settings",
        lambda: Settings.model_construct(max_supervisor_steps=12),
    )
    result = supervisor_node(
        {"messages": [HumanMessage(content="Write fibonacci")], "research_notes": "use recursion"},
        {},
    )
    assert result["next_action"] == "coder"
    assert result["step_count"] == 1


def test_researcher_writes_notes(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_team.nodes.researcher.get_llm",
        lambda: SimpleNamespace(invoke=lambda _msgs: AIMessage(content="use recursion")),
    )
    result = researcher_node({"messages": [HumanMessage(content="fib")]})
    assert result["research_notes"] == "use recursion"
    assert result["messages"][0].content == "use recursion"


def test_coder_strips_fences(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_team.nodes.coder.get_llm",
        lambda: SimpleNamespace(
            invoke=lambda _msgs: AIMessage(content="```python\ndef foo():\n    return 1\n```")
        ),
    )
    result = coder_node({"messages": [HumanMessage(content="foo")]})
    assert result["current_code"] == "def foo():\n    return 1"
    assert "```" not in result["current_code"]


def test_editor_writes_notes(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_team.nodes.editor.get_llm",
        lambda: SimpleNamespace(invoke=lambda _msgs: AIMessage(content="looks fine")),
    )
    result = editor_node(
        {"messages": [HumanMessage(content="foo")], "current_code": "def foo(): pass"}
    )
    assert result["editor_notes"] == "looks fine"


def test_human_review_approves() -> None:
    result = human_review_node({"feedback": "APPROVE", "messages": []}, {})
    assert result["next_action"] == "finish"
    assert result["feedback"] == "APPROVE"


def test_human_review_rejects_with_comment() -> None:
    result = human_review_node({"feedback": "add type hints", "messages": []}, {})
    assert result["next_action"] == "supervisor"
    assert result["feedback"] == "add type hints"


def test_human_review_empty_feedback_requests_changes() -> None:
    result = human_review_node({"messages": []}, {})
    assert result["next_action"] == "supervisor"
    assert "Changes requested" in result["feedback"]


def test_human_review_auto_approve_from_config() -> None:
    result = human_review_node(
        {"feedback": "ignored", "messages": []},
        {"configurable": {"auto_approve": True}},
    )
    assert result["next_action"] == "finish"
    assert result["feedback"] == "APPROVE"


def test_human_review_auto_approve_from_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        "agent_team.nodes.human_review.get_settings",
        lambda: Settings.model_construct(auto_approve=True),
    )
    result = human_review_node({"messages": []}, {})
    assert result["next_action"] == "finish"
    assert result["feedback"] == "APPROVE"
