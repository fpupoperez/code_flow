"""Compile the supervisor multi-agent graph.

The module-level ``graph`` (aliased as ``app``) is what LangGraph Server loads
from langgraph.json. It is compiled *without* a checkpointer so the platform
can inject Postgres persistence.

Tracing defaults to Langfuse (MIT, self-hostable) via ``attach_tracing``.
MLflow or LangSmith is used only when Langfuse keys are absent.
See ``agent_team.tracing``.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent_team.nodes.coder import coder_node
from agent_team.nodes.editor import editor_node
from agent_team.nodes.human_review import human_review_node, route_after_human
from agent_team.nodes.researcher import researcher_node
from agent_team.nodes.review_publisher import review_publisher_node
from agent_team.nodes.supervisor import route_next, supervisor_node
from agent_team.state import AgentState
from agent_team.tracing import attach_tracing, configure_tracing


def build_graph(*, checkpointer=None) -> CompiledStateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("coder", coder_node)
    workflow.add_node("editor", editor_node)
    workflow.add_node("review_publisher", review_publisher_node)
    workflow.add_node("human_review", human_review_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "researcher": "researcher",
            "coder": "coder",
            "editor": "editor",
            "human_review": "review_publisher",
        },
    )
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("coder", "supervisor")
    workflow.add_edge("editor", "supervisor")
    workflow.add_edge("review_publisher", "human_review")
    workflow.add_conditional_edges(
        "human_review",
        route_after_human,
        {
            "finish": END,
            "supervisor": "supervisor",
        },
    )

    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_review"],
    )


def build_local_graph() -> CompiledStateGraph:
    """In-memory checkpointer for the terminal CLI."""
    return build_graph(checkpointer=InMemorySaver())


configure_tracing()
# Server export: bind Langfuse callbacks here because LangGraph Server invokes
# this object directly and never goes through the CLI merge helper.
graph = attach_tracing(build_graph())
app = graph
