from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_team import graph as graph_mod
from agent_team.graph import build_graph, build_local_graph


def test_compiled_graph_interrupts_before_human_review() -> None:
    compiled = build_graph()
    assert "human_review" in compiled.interrupt_before_nodes


def test_server_graph_has_no_checkpointer() -> None:
    compiled = build_graph()
    assert compiled.checkpointer is None


def test_local_graph_uses_in_memory_checkpointer() -> None:
    compiled = build_local_graph()
    assert compiled.checkpointer is not None


def test_local_graph_pauses_before_human_review(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_mod,
        "supervisor_node",
        lambda state, config: {"next_action": "human_review", "step_count": 1},
    )
    monkeypatch.setattr(graph_mod, "review_publisher_node", lambda state, config: {})

    compiled = graph_mod.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-hitl"}}
    compiled.invoke({"messages": [HumanMessage(content="Write fibonacci")]}, config)

    snapshot = compiled.get_state(config)
    assert "human_review" in snapshot.next
    assert snapshot.values["step_count"] == 1


def test_approve_resumes_and_finishes(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_mod,
        "supervisor_node",
        lambda state, config: {"next_action": "human_review", "step_count": 1},
    )
    monkeypatch.setattr(graph_mod, "review_publisher_node", lambda state, config: {})

    compiled = graph_mod.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-approve"}}
    compiled.invoke({"messages": [HumanMessage(content="do it")]}, config)
    compiled.update_state(config, {"feedback": "APPROVE"})
    compiled.invoke(None, config)

    snapshot = compiled.get_state(config)
    assert snapshot.next == ()
    assert snapshot.values["next_action"] == "finish"
    assert snapshot.values["feedback"] == "APPROVE"


def test_reject_returns_to_supervisor_then_pauses_again(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_mod,
        "supervisor_node",
        lambda state, config: {"next_action": "human_review", "step_count": 1},
    )
    monkeypatch.setattr(graph_mod, "review_publisher_node", lambda state, config: {})

    compiled = graph_mod.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-reject"}}
    compiled.invoke({"messages": [HumanMessage(content="do it")]}, config)
    compiled.update_state(config, {"feedback": "add tests"})
    compiled.invoke(None, config)

    snapshot = compiled.get_state(config)
    assert "human_review" in snapshot.next
    assert snapshot.values["feedback"] == "add tests"


def test_specialist_cycle_then_review(monkeypatch) -> None:
    calls = {"supervisor": 0}

    def fake_supervisor(state, config):
        calls["supervisor"] += 1
        if calls["supervisor"] == 1:
            return {"next_action": "researcher", "step_count": 1}
        if calls["supervisor"] == 2:
            return {"next_action": "coder", "step_count": 2}
        if calls["supervisor"] == 3:
            return {"next_action": "editor", "step_count": 3}
        return {"next_action": "human_review", "step_count": 4}

    monkeypatch.setattr(graph_mod, "supervisor_node", fake_supervisor)
    monkeypatch.setattr(
        graph_mod, "researcher_node", lambda state: {"research_notes": "use recursion"}
    )
    monkeypatch.setattr(
        graph_mod, "coder_node", lambda state: {"current_code": "def fib(n): return n"}
    )
    monkeypatch.setattr(graph_mod, "editor_node", lambda state: {"editor_notes": "ok"})
    monkeypatch.setattr(graph_mod, "review_publisher_node", lambda state, config: {})

    compiled = graph_mod.build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t-cycle"}}
    compiled.invoke({"messages": [HumanMessage(content="fib")]}, config)

    snapshot = compiled.get_state(config)
    assert "human_review" in snapshot.next
    assert snapshot.values["research_notes"] == "use recursion"
    assert snapshot.values["current_code"] == "def fib(n): return n"
    assert snapshot.values["editor_notes"] == "ok"
    assert calls["supervisor"] == 4
