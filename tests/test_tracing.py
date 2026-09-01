import os
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock

from agent_team.settings import Settings
from agent_team.tracing import (
    attach_tracing,
    configure_tracing,
    flush_tracing,
    merge_invoke_config,
    resolve_tracing_backend,
)


def _settings(**kwargs) -> Settings:
    # Bypass dotenv so a developer's .env cannot flip the selected backend.
    return Settings.model_construct(**kwargs)


def test_langfuse_wins_when_all_backends_are_configured() -> None:
    settings = _settings(
        langfuse_enabled=True,
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        mlflow_enabled=True,
        mlflow_tracking_uri="http://localhost:5000",
        langsmith_tracing=True,
        langsmith_api_key="ls-test",
    )
    assert resolve_tracing_backend(settings) == "langfuse"


def test_mlflow_is_used_when_langfuse_keys_are_missing() -> None:
    settings = _settings(
        langfuse_enabled=True,
        langfuse_public_key="",
        langfuse_secret_key="",
        mlflow_enabled=True,
        mlflow_tracking_uri="http://localhost:5000",
        langsmith_tracing=True,
        langsmith_api_key="ls-test",
    )
    assert resolve_tracing_backend(settings) == "mlflow"


def test_langsmith_is_used_when_langfuse_and_mlflow_are_missing() -> None:
    settings = _settings(
        langfuse_enabled=True,
        langfuse_public_key="",
        langfuse_secret_key="",
        mlflow_enabled=True,
        mlflow_tracking_uri="",
        langsmith_tracing=True,
        langsmith_api_key="ls-test",
    )
    assert resolve_tracing_backend(settings) == "langsmith"


def test_no_tracing_without_credentials() -> None:
    settings = _settings(
        langfuse_public_key="",
        langfuse_secret_key="",
        mlflow_tracking_uri="",
        langsmith_tracing=True,
        langsmith_api_key="",
    )
    assert resolve_tracing_backend(settings) == "none"


def test_langfuse_can_be_disabled_to_fall_back_to_mlflow() -> None:
    settings = _settings(
        langfuse_enabled=False,
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        mlflow_enabled=True,
        mlflow_tracking_uri="file:./mlruns",
        langsmith_tracing=True,
        langsmith_api_key="ls-test",
    )
    assert resolve_tracing_backend(settings) == "mlflow"


def test_mlflow_can_be_disabled_to_fall_back_to_langsmith() -> None:
    settings = _settings(
        langfuse_enabled=False,
        langfuse_public_key="",
        langfuse_secret_key="",
        mlflow_enabled=False,
        mlflow_tracking_uri="http://localhost:5000",
        langsmith_tracing=True,
        langsmith_api_key="ls-test",
    )
    assert resolve_tracing_backend(settings) == "langsmith"


def test_merge_invoke_config_is_noop_without_langfuse(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    monkeypatch.setattr(tracing_mod, "configure_tracing", lambda: "none")
    monkeypatch.setattr(tracing_mod, "tracing_callbacks", lambda: [])
    config = merge_invoke_config({"configurable": {"thread_id": "t1"}})
    assert "callbacks" not in config
    assert config["configurable"]["thread_id"] == "t1"
    tracing_mod.reset_tracing_for_tests()


def test_merge_invoke_config_adds_langfuse_session(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    monkeypatch.setattr(tracing_mod, "configure_tracing", lambda: "langfuse")
    monkeypatch.setattr(tracing_mod, "tracing_callbacks", lambda: ["handler"])
    config = merge_invoke_config({"configurable": {"thread_id": "t1"}})
    assert config["callbacks"] == ["handler"]
    assert config["run_name"] == "agent-team"
    assert config["metadata"]["thread_id"] == "t1"
    assert config["metadata"]["langfuse_session_id"] == "t1"
    tracing_mod.reset_tracing_for_tests()


def test_merge_invoke_config_mlflow_has_thread_but_not_langfuse_session(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    monkeypatch.setattr(tracing_mod, "configure_tracing", lambda: "mlflow")
    monkeypatch.setattr(tracing_mod, "tracing_callbacks", lambda: [])
    config = merge_invoke_config({"configurable": {"thread_id": "t2"}})
    assert config["metadata"]["thread_id"] == "t2"
    assert "langfuse_session_id" not in config["metadata"]
    tracing_mod.reset_tracing_for_tests()


def test_attach_tracing_noop_without_callbacks(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    monkeypatch.setattr(tracing_mod, "tracing_callbacks", lambda: [])
    compiled = object()
    assert attach_tracing(compiled) is compiled


def test_attach_tracing_binds_callbacks(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    compiled = Mock()
    compiled.with_config.return_value = "bound"
    monkeypatch.setattr(tracing_mod, "tracing_callbacks", lambda: ["cb"])
    assert attach_tracing(compiled) == "bound"
    compiled.with_config.assert_called_once_with({"callbacks": ["cb"], "run_name": "agent-team"})


def test_configure_tracing_is_idempotent(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    monkeypatch.setattr(tracing_mod, "resolve_tracing_backend", lambda _settings: "none")
    assert configure_tracing(_settings()) == "none"
    assert configure_tracing(_settings(langsmith_tracing=True, langsmith_api_key="x")) == "none"
    tracing_mod.reset_tracing_for_tests()


def test_configure_tracing_failure_becomes_none(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    monkeypatch.setattr(tracing_mod, "resolve_tracing_backend", lambda _settings: "langfuse")
    monkeypatch.setattr(
        tracing_mod, "_configure_langfuse", Mock(side_effect=RuntimeError("boom"))
    )
    assert configure_tracing(_settings()) == "none"
    tracing_mod.reset_tracing_for_tests()


def test_configure_langsmith_sets_env(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    monkeypatch.setattr(tracing_mod, "resolve_tracing_backend", lambda _settings: "langsmith")
    settings = _settings(
        langsmith_tracing=True,
        langsmith_api_key="ls-test",
        langsmith_project="proj",
    )
    assert configure_tracing(settings) == "langsmith"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test"
    assert os.environ["LANGSMITH_PROJECT"] == "proj"
    tracing_mod.reset_tracing_for_tests()


def test_configure_langfuse_disables_langsmith(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    fake = types.ModuleType("langfuse")
    fake.Langfuse = Mock()
    fake.get_client = Mock()
    monkeypatch.setitem(sys.modules, "langfuse", fake)
    monkeypatch.setattr(tracing_mod, "resolve_tracing_backend", lambda _settings: "langfuse")
    settings = _settings(
        langfuse_enabled=True,
        langfuse_public_key="pk",
        langfuse_secret_key="sk",
        langfuse_host="http://langfuse:3000",
    )
    assert configure_tracing(settings) == "langfuse"
    assert os.environ["LANGSMITH_TRACING"] == "false"
    fake.Langfuse.assert_called_once()
    tracing_mod.reset_tracing_for_tests()


def test_configure_mlflow_autolog(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    tracing_mod.reset_tracing_for_tests()
    fake_ml = types.ModuleType("mlflow")
    fake_ml.set_tracking_uri = Mock()
    fake_ml.set_experiment = Mock()
    fake_ml.langchain = SimpleNamespace(autolog=Mock())
    monkeypatch.setitem(sys.modules, "mlflow", fake_ml)
    monkeypatch.setattr(tracing_mod, "resolve_tracing_backend", lambda _settings: "mlflow")
    settings = _settings(
        mlflow_enabled=True,
        mlflow_tracking_uri="http://mlflow:5000",
        mlflow_experiment_name="agent-team",
    )
    assert configure_tracing(settings) == "mlflow"
    fake_ml.set_tracking_uri.assert_called_once_with("http://mlflow:5000")
    fake_ml.langchain.autolog.assert_called_once_with(log_traces=True, run_tracer_inline=True)
    tracing_mod.reset_tracing_for_tests()


def test_flush_tracing_langfuse(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    flush = Mock()
    fake = types.ModuleType("langfuse")
    fake.get_client = lambda: SimpleNamespace(flush=flush)
    monkeypatch.setitem(sys.modules, "langfuse", fake)
    monkeypatch.setattr(tracing_mod, "configure_tracing", lambda: "langfuse")
    flush_tracing()
    flush.assert_called_once()


def test_flush_tracing_mlflow(monkeypatch) -> None:
    from agent_team import tracing as tracing_mod

    flush = Mock()
    fake_ml = types.ModuleType("mlflow")
    fake_ml.flush_trace = flush
    monkeypatch.setitem(sys.modules, "mlflow", fake_ml)
    monkeypatch.setattr(tracing_mod, "configure_tracing", lambda: "mlflow")
    flush_tracing()
    flush.assert_called_once()
