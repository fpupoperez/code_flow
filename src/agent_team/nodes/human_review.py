"""Human review node. Execution pauses before this node via interrupt_before."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.runnables import RunnableConfig

from agent_team.settings import get_settings
from agent_team.state import AgentState
from agent_team.utils import is_approved

logger = logging.getLogger(__name__)


def human_review_node(state: AgentState, config: RunnableConfig) -> dict:
    """Consume human feedback injected while the graph was paused.

    The graph is compiled with ``interrupt_before=["human_review"]``. A Slack
    click or the CLI writes ``feedback`` into the checkpoint, then the run is
    resumed so this node can decide whether to finish or send work back.
    """
    configurable = (config or {}).get("configurable") or {}
    auto_approve = bool(configurable.get("auto_approve", get_settings().auto_approve))
    feedback = "APPROVE" if auto_approve else (state.get("feedback") or "")

    if is_approved(feedback):
        logger.info("Human approved the artifact")
        return {"feedback": feedback, "next_action": "finish"}

    logger.info("Human requested changes; returning to supervisor")
    return {
        "feedback": feedback or "Changes requested without comments.",
        "next_action": "supervisor",
    }


def route_after_human(state: AgentState) -> Literal["finish", "supervisor"]:
    if state.get("next_action") == "finish":
        return "finish"
    return "supervisor"
