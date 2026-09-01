"""Researcher specialist."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent_team.llm import get_llm
from agent_team.prompts import RESEARCHER_SYSTEM
from agent_team.state import AgentState
from agent_team.utils import original_request

logger = logging.getLogger(__name__)


def researcher_node(state: AgentState) -> dict:
    logger.info("Researcher working")
    request = original_request(state)
    feedback = state.get("feedback") or "None"
    prompt = (
        f"Research the requirements for:\n{request}\n\n"
        f"Human feedback to incorporate: {feedback}\n"
        "Provide concise technical details, constraints, and a suggested implementation plan."
    )
    response = get_llm().invoke(
        [SystemMessage(content=RESEARCHER_SYSTEM), HumanMessage(content=prompt)]
    )
    return {"research_notes": response.content, "messages": [response]}
