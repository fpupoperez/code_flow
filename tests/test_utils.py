from langchain_core.messages import AIMessage, HumanMessage

from agent_team.utils import is_approved, message_text, original_request, strip_code_fences


def test_message_text_from_string_dict_and_blocks() -> None:
    assert message_text(None) == ""
    assert message_text("plain") == "plain"
    assert message_text({"content": "from-dict"}) == "from-dict"
    assert (
        message_text(
            {
                "content": [
                    "lead",
                    {"type": "text", "text": "block"},
                    {"type": "image", "url": "x"},
                ]
            }
        )
        == "lead\nblock"
    )
    assert message_text(HumanMessage(content="human")) == "human"


def test_original_request_uses_first_human_not_last_ai() -> None:
    state = {
        "messages": [
            HumanMessage(content="Write fibonacci"),
            AIMessage(content="Looks ready for review."),
        ]
    }
    assert original_request(state) == "Write fibonacci"


def test_original_request_accepts_role_user_dicts() -> None:
    state = {"messages": [{"role": "user", "content": "from role"}]}
    assert original_request(state) == "from role"


def test_original_request_empty() -> None:
    assert original_request({"messages": []}) == "No context."


def test_strip_code_fences_plain_and_fenced() -> None:
    assert strip_code_fences("def x():\n    return 1") == "def x():\n    return 1"
    assert strip_code_fences("```\njust text\n```") == "just text"


def test_is_approved_tokens() -> None:
    assert is_approved("LGTM")
    assert is_approved("  approved  ")
    assert not is_approved(None)
