"""Coder specialist."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent_team.llm import get_llm
from agent_team.prompts import CODER_SYSTEM
from agent_team.state import AgentState
from agent_team.utils import original_request, strip_code_fences

logger = logging.getLogger(__name__)


def coder_node(state: AgentState) -> dict:
    logger.info("Coder working")
    notes = state.get("research_notes") or "No research notes yet."
    feedback = state.get("feedback") or "None"
    existing = state.get("current_code") or ""
    prompt = (
        f"User request:\n{original_request(state)}\n\n"
        f"Research notes:\n{notes}\n\n"
        f"Existing code (may be empty):\n{existing}\n\n"
        f"Feedback to address:\n{feedback}"
    )
    response = get_llm().invoke(
        [SystemMessage(content=CODER_SYSTEM), HumanMessage(content=prompt)]
    )
    code = strip_code_fences(str(response.content))
    return {"current_code": code, "messages": [response]}
