from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

from evals.evaluate_agents import (
    Grade,
    _ensure_dataset,
    code_correctness_evaluator,
    cost_latency_evaluator,
    routing_efficiency_evaluator,
    target_graph_runner,
)


def test_routing_efficiency_ideal() -> None:
    run = SimpleNamespace(
        child_runs=[
            SimpleNamespace(name="researcher"),
            SimpleNamespace(name="coder"),
            SimpleNamespace(name="editor"),
            SimpleNamespace(name="supervisor"),
        ]
    )
    result = routing_efficiency_evaluator(run, None)
    assert result["key"] == "routing_efficiency"
    assert result["score"] == 1.0


def test_routing_efficiency_excess_steps() -> None:
    run = SimpleNamespace(child_runs=[SimpleNamespace(name="researcher") for _ in range(8)])
    result = routing_efficiency_evaluator(run, None)
    assert result["score"] == 0.0


def test_routing_efficiency_partial_penalty() -> None:
    run = SimpleNamespace(
        child_runs=[
            SimpleNamespace(name="researcher"),
            SimpleNamespace(name="coder"),
            SimpleNamespace(name="editor"),
            SimpleNamespace(name="coder"),
        ]
    )
    result = routing_efficiency_evaluator(run, None)
    assert result["score"] == 0.8


def test_cost_latency_fast_and_cheap() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0)
    run = SimpleNamespace(
        start_time=start,
        end_time=start + timedelta(seconds=9),
        total_tokens=0,
    )
    result = cost_latency_evaluator(run, None)
    assert result["key"] == "cost_latency_efficiency"
    assert result["score"] > 0.8


def test_cost_latency_missing_times_scores_as_zero_elapsed() -> None:
    result = cost_latency_evaluator(SimpleNamespace(), None)
    assert result["score"] == 1.0


def test_code_correctness_uses_judge(monkeypatch) -> None:
    class Graded:
        def invoke(self, _prompt):
            return Grade(score=0.8, comment="close")

    class FakeJudge:
        def with_structured_output(self, *_args, **_kwargs):
            return Graded()

    monkeypatch.setattr("evals.evaluate_agents._judge", lambda: FakeJudge())
    run = SimpleNamespace(outputs={"output": "def fibonacci(): pass"})
    example = SimpleNamespace(outputs={"expected": "def fibonacci"})
    result = code_correctness_evaluator(run, example)
    assert result == {"key": "code_quality", "score": 0.8, "comment": "close"}


def test_code_correctness_falls_back_to_substring(monkeypatch) -> None:
    class Boom:
        def invoke(self, _prompt):
            raise ValueError("bad json")

    class FakeJudge:
        def with_structured_output(self, *_args, **_kwargs):
            return Boom()

    monkeypatch.setattr("evals.evaluate_agents._judge", lambda: FakeJudge())
    run = SimpleNamespace(outputs={"output": "def fibonacci(): pass"})
    example = SimpleNamespace(outputs={"expected": "def fibonacci"})
    result = code_correctness_evaluator(run, example)
    assert result["score"] == 1.0
    assert "Fallback" in result["comment"]

    miss = SimpleNamespace(outputs={"output": "def other(): pass"})
    result = code_correctness_evaluator(miss, example)
    assert result["score"] == 0.0


def test_ensure_dataset_skips_when_present() -> None:
    client = Mock()
    client.list_datasets.return_value = [Mock()]
    _ensure_dataset(client)
    client.create_dataset.assert_not_called()


def test_ensure_dataset_creates_examples() -> None:
    client = Mock()
    client.list_datasets.return_value = []
    client.create_dataset.return_value = SimpleNamespace(id="ds-1")
    _ensure_dataset(client)
    client.create_dataset.assert_called_once()
    client.create_examples.assert_called_once()
    kwargs = client.create_examples.call_args.kwargs
    assert kwargs["dataset_id"] == "ds-1"
    assert len(kwargs["examples"]) == 2


def test_target_graph_runner_uses_local(monkeypatch) -> None:
    monkeypatch.setenv("USE_LOCAL_GRAPH", "1")
    monkeypatch.setattr(
        "evals.evaluate_agents._run_local", lambda _inputs: {"output": "def x(): pass"}
    )
    assert target_graph_runner({"messages": []}) == {"output": "def x(): pass"}
