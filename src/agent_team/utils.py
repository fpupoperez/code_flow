"""Helpers shared by graph nodes."""

from langchain_core.messages import BaseMessage, HumanMessage

from agent_team.state import AgentState

APPROVAL_TOKENS = ("APPROVE", "APPROVED", "LGTM")


def message_text(message: BaseMessage | dict | str | None) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def original_request(state: AgentState) -> str:
    for message in state.get("messages", []):
        if isinstance(message, HumanMessage) or getattr(message, "type", None) == "human":
            text = message_text(message)
            if text:
                return text
        if isinstance(message, dict) and message.get("role") == "user":
            text = message_text(message)
            if text:
                return text
    return "No context."


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def is_approved(feedback: str | None) -> bool:
    if not feedback:
        return False
    upper = feedback.strip().upper()
    return any(token in upper for token in APPROVAL_TOKENS)
