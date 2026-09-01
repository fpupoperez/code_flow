import sys
from types import SimpleNamespace

import pytest

from agent_team.cli import main, run_cli


class FakeGraph:
    def __init__(self) -> None:
        self.feedback: str | None = None
        self.phase = "paused"

    def invoke(self, _payload, _config) -> None:
        if self.feedback == "APPROVE" or (
            self.feedback and self.feedback.upper() in {"APPROVE", "APPROVED", "LGTM"}
        ):
            self.phase = "done"

    def get_state(self, _config):
        values = {"current_code": "def x(): pass", "editor_notes": "ok"}
        if self.phase == "done":
            return SimpleNamespace(next=(), values=values)
        return SimpleNamespace(next=("human_review",), values=values)

    def update_state(self, _config, values) -> None:
        self.feedback = values["feedback"]


def test_run_cli_finishes_on_approve(monkeypatch) -> None:
    monkeypatch.setattr("agent_team.cli.build_local_graph", FakeGraph)
    monkeypatch.setattr("agent_team.cli.merge_invoke_config", lambda config: config)
    monkeypatch.setattr("agent_team.cli.flush_tracing", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: "APPROVE")

    result = run_cli("do it", thread_id="t-cli")
    assert result["current_code"] == "def x(): pass"


def test_run_cli_revision_then_approve(monkeypatch) -> None:
    answers = iter(["needs tests", "APPROVE"])
    monkeypatch.setattr("agent_team.cli.build_local_graph", FakeGraph)
    monkeypatch.setattr("agent_team.cli.merge_invoke_config", lambda config: config)
    monkeypatch.setattr("agent_team.cli.flush_tracing", lambda: None)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    result = run_cli("do it", thread_id="t-cli-2")
    assert "def x" in result["current_code"]


def test_main_uses_argv_prompt(monkeypatch) -> None:
    called: dict[str, str] = {}
    monkeypatch.setattr(sys, "argv", ["agent-team", "build", "foo"])
    monkeypatch.setattr("agent_team.cli.run_cli", lambda prompt: called.setdefault("prompt", prompt))
    main()
    assert called["prompt"] == "build foo"


def test_main_exits_without_prompt(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["agent-team"])
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
