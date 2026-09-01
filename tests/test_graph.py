from agent_team.graph import build_graph
from agent_team.nodes.human_review import route_after_human
from agent_team.utils import is_approved, strip_code_fences
from slack_gateway.payloads import ReviewDecision, decision_from_slack, parse_review_payload
from slack_gateway.signatures import verify_slack_signature


def test_graph_contains_expected_nodes() -> None:
    compiled = build_graph()
    nodes = set(compiled.get_graph().nodes)
    for name in (
        "supervisor",
        "researcher",
        "coder",
        "editor",
        "review_publisher",
        "human_review",
    ):
        assert name in nodes


def test_route_after_human() -> None:
    assert route_after_human({"messages": [], "next_action": "finish"}) == "finish"
    assert route_after_human({"messages": [], "next_action": "supervisor"}) == "supervisor"
    assert route_after_human({"messages": []}) == "supervisor"


def test_is_approved() -> None:
    assert is_approved("APPROVE")
    assert is_approved("please APPROVED thanks")
    assert not is_approved("needs more tests")
    assert not is_approved("")


def test_strip_code_fences() -> None:
    raw = "```python\nprint('hi')\n```"
    assert strip_code_fences(raw) == "print('hi')"


def test_parse_approve_payload() -> None:
    payload = {
        "actions": [{"action_id": "approve_btn", "value": "thread-1"}],
        "state": {"values": {}},
    }
    decision = parse_review_payload(payload)
    assert isinstance(decision, ReviewDecision)
    assert decision.action == "approve"
    assert decision.feedback == "APPROVE"
    assert decision.thread_id == "thread-1"


def test_parse_reject_payload_with_comment() -> None:
    payload = {
        "actions": [{"action_id": "reject_btn", "value": "thread-2"}],
        "state": {
            "values": {
                "feedback_block": {
                    "feedback_input": {
                        "type": "plain_text_input",
                        "value": "Add type hints",
                    }
                }
            }
        },
    }
    decision = parse_review_payload(payload)
    assert isinstance(decision, ReviewDecision)
    assert decision.action == "reject"
    assert decision.feedback == "Add type hints"


def test_decision_from_slack_includes_reviewer_context() -> None:
    payload = {
        "actions": [{"action_id": "reject_btn", "value": "thread-2"}],
        "user": {"id": "U1", "username": "ana"},
        "channel": {"id": "C9"},
        "response_url": "https://hooks.slack.com/actions/T/x",
        "trigger_id": "trig-1",
        "state": {
            "values": {
                "feedback_block": {
                    "feedback_input": {
                        "type": "plain_text_input",
                        "value": "Add type hints",
                    }
                }
            }
        },
    }
    decision = parse_review_payload(payload)
    assert isinstance(decision, ReviewDecision)
    event = decision_from_slack(decision, payload, assistant_id="agent_team")
    assert event.event_type == "human_review_submitted"
    assert event.thread_id == "thread-2"
    assert event.action == "reject"
    assert event.feedback == "Add type hints"
    assert event.slack_user_id == "U1"
    assert event.slack_username == "ana"
    assert event.slack_channel == "C9"
    assert event.slack_response_url.startswith("https://hooks.slack.com")
    assert event.slack_context["user"]["id"] == "U1"


def test_slack_signature_roundtrip() -> None:
    secret = "test-secret"
    body = b"payload=%7B%7D"
    timestamp = "1000000000"
    import hashlib
    import hmac

    digest = hmac.new(
        secret.encode(), b"v0:" + timestamp.encode() + b":" + body, hashlib.sha256
    ).hexdigest()
    assert verify_slack_signature(
        signing_secret=secret,
        body=body,
        timestamp=timestamp,
        signature=f"v0={digest}",
        max_age_seconds=10**12,
    )
    assert not verify_slack_signature(
        signing_secret=secret,
        body=body,
        timestamp=timestamp,
        signature="v0=deadbeef",
        max_age_seconds=10**12,
    )
