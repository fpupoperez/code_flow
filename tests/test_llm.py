import sys
import types

from agent_team import llm as llm_mod
from agent_team.settings import Settings


def test_get_llm_uses_settings(monkeypatch) -> None:
    llm_mod.get_llm.cache_clear()
    monkeypatch.setattr(
        "agent_team.llm.get_settings",
        lambda: Settings.model_construct(openai_model="gpt-4o-mini", openai_api_key="sk-test"),
    )

    captured: dict = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_mod = types.ModuleType("langchain_openai")
    fake_mod.ChatOpenAI = FakeChat
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_mod)

    result = llm_mod.get_llm()
    assert isinstance(result, FakeChat)
    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0
    assert captured["api_key"] == "sk-test"
    llm_mod.get_llm.cache_clear()
