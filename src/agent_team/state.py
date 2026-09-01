"""Shared graph state for the supervisor team."""

from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

AgentName = Literal["researcher", "coder", "editor", "human_review"]
NextAction = Literal["researcher", "coder", "editor", "human_review", "finish", "supervisor"]


class AgentState(TypedDict):
    """Central state shared by every node in the graph."""

    messages: Annotated[list[AnyMessage], add_messages]
    research_notes: NotRequired[str]
    current_code: NotRequired[str]
    editor_notes: NotRequired[str]
    feedback: NotRequired[str]
    next_action: NotRequired[NextAction]
    step_count: NotRequired[int]


class GraphConfig(TypedDict, total=False):
    """Per-run configuration (thread_id, eval flags, etc.)."""

    thread_id: str
    auto_approve: bool
