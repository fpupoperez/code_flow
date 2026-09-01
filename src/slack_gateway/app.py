"""FastAPI webhook that publishes Slack review actions onto NATS JetStream."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_team.settings import Settings, get_settings
from review_events.jetstream import ensure_stream, jetstream_connection, publish_event
from review_events.models import HumanReviewSubmitted
from slack_gateway.payloads import IgnoredAction, decision_from_slack, parse_review_payload
from slack_gateway.signatures import verify_slack_signature
from telemetry.otel import configure_otel, instrument_fastapi

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_otel(default_service_name="slack-gateway", settings=settings)
    app = FastAPI(
        title="LangGraph Slack Gateway",
        description="Queues Slack Block Kit review actions onto NATS JetStream.",
        version="1.0.0",
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/slack/interactive")
    async def slack_interactive(request: Request):
        raw_body = await request.body()
        if settings.slack_signing_secret:
            valid = verify_slack_signature(
                signing_secret=settings.slack_signing_secret,
                body=raw_body,
                timestamp=request.headers.get("X-Slack-Request-Timestamp", ""),
                signature=request.headers.get("X-Slack-Signature", ""),
            )
            if not valid:
                raise HTTPException(status_code=401, detail="Invalid Slack signature")

        form = await request.form()
        payload_raw = form.get("payload")
        if not payload_raw:
            raise HTTPException(status_code=400, detail="Missing Slack payload")

        try:
            payload = json.loads(str(payload_raw))
            decision = parse_review_payload(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if isinstance(decision, IgnoredAction):
            return JSONResponse({"status": "ignored_action", "action_id": decision.action_id})

        event = decision_from_slack(
            decision,
            payload,
            assistant_id=settings.langgraph_assistant_id,
        )
        try:
            await publish_feedback_event(event, settings)
        except Exception as exc:
            logger.exception("Failed to publish Slack feedback for thread %s", event.thread_id)
            raise HTTPException(status_code=503, detail="Failed to queue review decision") from exc

        return JSONResponse(
            {
                "response_type": "ephemeral",
                "text": "Choice submitted. Agents are resuming work now.",
            }
        )

    instrument_fastapi(app)
    return app


async def publish_feedback_event(event: HumanReviewSubmitted, settings: Settings) -> None:
    if not settings.nats_enabled:
        raise RuntimeError("NATS_ENABLED must be true for the Slack gateway")

    async with jetstream_connection(settings.nats_url, name="slack-gateway") as (_nc, js):
        await ensure_stream(
            js,
            stream=settings.nats_stream,
            subject=settings.nats_feedback_subject,
        )
        await publish_event(js, subject=settings.nats_feedback_subject, event=event)


app = create_app()
