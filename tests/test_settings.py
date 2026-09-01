from agent_team.settings import Settings


def test_langfuse_host_alias() -> None:
    settings = Settings.model_validate({"LANGFUSE_BASE_URL": "http://langfuse:3000"})
    assert settings.langfuse_host == "http://langfuse:3000"


def test_nats_queue_durable_alias() -> None:
    settings = Settings.model_validate({"NATS_DURABLE": "from-durable"})
    assert settings.nats_queue == "from-durable"


def test_langsmith_legacy_aliases() -> None:
    settings = Settings.model_validate(
        {
            "LANGCHAIN_TRACING_V2": True,
            "LANGCHAIN_API_KEY": "ls-legacy",
            "LANGCHAIN_PROJECT": "legacy-project",
        }
    )
    assert settings.langsmith_tracing is True
    assert settings.langsmith_api_key == "ls-legacy"
    assert settings.langsmith_project == "legacy-project"
