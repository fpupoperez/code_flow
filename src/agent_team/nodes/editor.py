"""Editor specialist."""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent_team.llm import get_llm
from agent_team.prompts import EDITOR_SYSTEM
from agent_team.state import AgentState
from agent_team.utils import original_request

logger = logging.getLogger(__name__)


def editor_node(state: AgentState) -> dict:
    logger.info("Editor working")
    code = state.get("current_code") or "No code generated yet."
    prompt = (
        f"User request:\n{original_request(state)}\n\n"
        f"Review this code for bugs or style improvements:\n{code}"
    )
    response = get_llm().invoke(
        [SystemMessage(content=EDITOR_SYSTEM), HumanMessage(content=prompt)]
    )
    notes = str(response.content)
    return {"editor_notes": notes, "messages": [response]}
