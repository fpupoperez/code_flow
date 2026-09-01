"""LangSmith evaluation harness for the supervisor team.

Production traces default to Langfuse (see agent_team.tracing). This script is
the LangSmith *alternative* for offline evals: it needs LANGSMITH_API_KEY even
when Langfuse is the live tracer.

Requires:
  - LANGSMITH_API_KEY
  - a running LangGraph Server (langgraph dev) unless you set USE_LOCAL_GRAPH=1
  - AUTO_APPROVE / configurable.auto_approve so HITL does not stall evals
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langsmith import Client, evaluate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DATASET_NAME = "Multi-Agent Code Generator Benchmark"
IDEAL_STEPS = 3
MAX_EXCESS_STEPS = 5
MAX_ALLOWED_TIME_SECONDS = 45.0
MAX_TOKEN_BUDGET = 15_000
AGENT_NODES = {"researcher", "coder", "editor"}


class Grade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    comment: str


def _judge():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0)


def _ensure_dataset(client: Client) -> None:
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Regression set for the researcher/coder/editor supervisor graph.",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {
                    "messages": [
                        {"role": "user", "content": "Write a Python function that computes Fibonacci numbers."}
                    ]
                },
                "outputs": {"expected": "def fibonacci"},
            },
            {
                "inputs": {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a Python function is_palindrome(text: str) -> bool.",
                        }
                    ]
                },
                "outputs": {"expected": "def is_palindrome"},
            },
        ],
    )


def _run_local(inputs: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    from agent_team.graph import build_local_graph
    from agent_team.tracing import flush_tracing, merge_invoke_config

    graph = build_local_graph()
    config = merge_invoke_config(
        {"configurable": {"thread_id": "eval-local", "auto_approve": True}}
    )
    messages = inputs.get("messages") or []
    if messages and isinstance(messages[0], dict):
        content = messages[0].get("content", "")
        payload = {"messages": [HumanMessage(content=content)]}
    else:
        payload = inputs
    result = graph.invoke(payload, config)
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        result = graph.invoke(None, config)
    flush_tracing()
    return {"output": result.get("current_code", "Execution failed")}


async def _run_remote(inputs: dict[str, Any]) -> dict[str, Any]:
    from langgraph_sdk import get_client

    client = get_client(url=os.getenv("LANGGRAPH_SERVER_URL", "http://localhost:8123"))
    thread = await client.threads.create()
    await client.runs.wait(
        thread_id=thread["thread_id"],
        assistant_id=os.getenv("LANGGRAPH_ASSISTANT_ID", "agent_team"),
        input=inputs,
        config={"configurable": {"auto_approve": True}},
    )
    final_state = await client.threads.get_state(thread["thread_id"])
    values = final_state.get("values") or {}
    return {"output": values.get("current_code", "Execution failed")}


def target_graph_runner(inputs: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("USE_LOCAL_GRAPH", "1") == "1":
        return _run_local(inputs)
    import asyncio

    return asyncio.run(_run_remote(inputs))


def code_correctness_evaluator(run, example) -> dict[str, Any]:
    agent_output = (run.outputs or {}).get("output", "")
    ground_truth = (example.outputs or {}).get("expected", "")
    structured = _judge().with_structured_output(Grade, method="json_schema")
    try:
        grade = structured.invoke(
            "You are an expert QA judge. Compare generated code with the requirement.\n"
            f"Generated code:\n{agent_output}\n\n"
            f"Target expectation:\n{ground_truth}\n"
            "Score 1.0 if the requirement is clearly met, else a partial score."
        )
        return {"key": "code_quality", "score": grade.score, "comment": grade.comment}
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("Judge fallback: %s", exc)
        hit = ground_truth.lower() in str(agent_output).lower()
        return {
            "key": "code_quality",
            "score": 1.0 if hit else 0.0,
            "comment": "Fallback substring match",
        }


def routing_efficiency_evaluator(run, example) -> dict[str, Any]:
    child_runs = getattr(run, "child_runs", None) or []
    total_steps = sum(1 for child in child_runs if getattr(child, "name", "") in AGENT_NODES)
    if total_steps <= IDEAL_STEPS:
        score = 1.0
        comment = f"Efficient routing: {total_steps} specialist steps."
    else:
        excess = total_steps - IDEAL_STEPS
        score = max(0.0, 1.0 - (excess / MAX_EXCESS_STEPS))
        comment = f"Suboptimal looping: {total_steps} specialist steps."
    return {"key": "routing_efficiency", "score": score, "comment": comment}


def cost_latency_evaluator(run, example) -> dict[str, Any]:
    start = getattr(run, "start_time", None)
    end = getattr(run, "end_time", None)
    elapsed = (end - start).total_seconds() if start and end else 0.0
    total_tokens = getattr(run, "total_tokens", None) or 0
    time_score = max(0.0, 1.0 - (elapsed / MAX_ALLOWED_TIME_SECONDS))
    token_score = max(0.0, 1.0 - (total_tokens / MAX_TOKEN_BUDGET))
    return {
        "key": "cost_latency_efficiency",
        "score": (time_score + token_score) / 2.0,
        "comment": f"Executed in {elapsed:.1f}s using {total_tokens:,} tokens.",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ls_client = Client()
    _ensure_dataset(ls_client)
    logger.info("Running LangSmith evaluation on %s", DATASET_NAME)
    evaluate(
        target_graph_runner,
        data=DATASET_NAME,
        evaluators=[
            code_correctness_evaluator,
            routing_efficiency_evaluator,
            cost_latency_evaluator,
        ],
        experiment_prefix="agent-team-v1",
    )


if __name__ == "__main__":
    main()
