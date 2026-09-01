"""Shared fixtures. Settings cache is cleared only when a test asks for it."""

from __future__ import annotations

import pytest

from agent_team.settings import get_settings


@pytest.fixture
def settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
