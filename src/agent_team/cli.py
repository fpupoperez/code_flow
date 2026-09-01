"""Interactive terminal runner for local human-in-the-loop review."""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from agent_team.graph import build_local_graph
from agent_team.tracing import flush_tracing, merge_invoke_config
from agent_team.utils import is_approved

logger = logging.getLogger(__name__)


def _print_state(values: dict[str, Any]) -> None:
    code = values.get("current_code") or "(none)"
    notes = values.get("editor_notes") or values.get("research_notes") or "(none)"
    print("\n--- Current artifact ---")
    print(code)
    print("\n--- Notes ---")
    print(notes)


def run_cli(prompt: str, *, thread_id: str | None = None) -> dict[str, Any]:
    graph = build_local_graph()
    config = merge_invoke_config(
        {
            "configurable": {"thread_id": thread_id or str(uuid.uuid4())},
            "recursion_limit": 50,
        }
    )
    try:
        graph.invoke({"messages": [HumanMessage(content=prompt)]}, config)

        while True:
            snapshot = graph.get_state(config)
            if not snapshot.next:
                print("\nWorkflow finished.")
                _print_state(snapshot.values)
                return snapshot.values

            if "human_review" in snapshot.next:
                _print_state(snapshot.values)
                feedback = input(
                    "\nHuman review — type APPROVE to finish, or describe required changes: "
                ).strip()
                graph.update_state(config, {"feedback": feedback or "APPROVE"})
                graph.invoke(None, config)
                if is_approved(feedback):
                    continue
                continue

            graph.invoke(None, config)
    finally:
        flush_tracing()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prompt = " ".join(sys.argv[1:]).strip() or input("What should the team build? ").strip()
    if not prompt:
        print("No prompt provided.")
        sys.exit(1)
    run_cli(prompt)


if __name__ == "__main__":
    main()
