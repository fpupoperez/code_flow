"""LLM factory."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from agent_team.settings import get_settings

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key or None,
    )
