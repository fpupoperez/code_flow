"""Supervisor router: structured output selects the next specialist."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from agent_team.llm import get_llm
from agent_team.prompts import SUPERVISOR_SYSTEM
from agent_team.settings import get_settings
from agent_team.state import AgentName, AgentState
from agent_team.utils import original_request

logger = logging.getLogger(__name__)


class Router(BaseModel):
    """Who should act next."""

    next_step: AgentName = Field(
        description="The next specialist to run, or human_review when ready for sign-off."
    )
    reason: str = Field(description="Short justification for the routing decision.")


def _state_brief(state: AgentState) -> str:
    feedback = state.get("feedback") or "None"
    notes = (state.get("research_notes") or "")[:500] or "(empty)"
    code = (state.get("current_code") or "")[:500] or "(empty)"
    editor = (state.get("editor_notes") or "")[:500] or "(empty)"
    return (
        f"Original request: {original_request(state)}\n"
        f"Step count: {state.get('step_count', 0)}\n"
        f"Human feedback: {feedback}\n"
        f"Research notes:\n{notes}\n"
        f"Current code:\n{code}\n"
        f"Editor notes:\n{editor}\n"
    )


def supervisor_node(state: AgentState, config: RunnableConfig) -> dict:
    settings = get_settings()
    step_count = int(state.get("step_count") or 0) + 1
    logger.info("Supervisor routing (step %s)", step_count)

    if step_count >= settings.max_supervisor_steps:
        logger.warning("Max supervisor steps reached; forcing human review")
        return {"next_action": "human_review", "step_count": step_count}

    llm = get_llm().with_structured_output(Router, method="json_schema")
    decision = llm.invoke(
        [
            SystemMessage(content=SUPERVISOR_SYSTEM),
            HumanMessage(content=_state_brief(state)),
            *list(state.get("messages") or []),
        ]
    )
    next_step = decision.next_step
    logger.info("Supervisor chose %s (%s)", next_step, decision.reason)
    return {"next_action": next_step, "step_count": step_count}


def route_next(state: AgentState) -> str:
    return state.get("next_action") or "researcher"
