"""Slack Web API stand-in for local HITL tests.

``slack_publisher`` posts Block Kit here via ``SLACK_API_BASE_URL``. The
emulator accepts the message (same ``ok: true`` shape as Slack) and, after a
short delay, POSTs a signed Approve payload to the gateway webhook so
``workflow_resumer`` can continue the graph.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from agent_team.settings import Settings, get_settings
from slack_emulator.payloads import build_approve_payload, thread_id_from_blocks
from slack_gateway.signatures import sign_slack_request

logger = logging.getLogger(__name__)


def _normalize_blocks(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, list):
        return value
    return []


async def _read_slack_body(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    form = await request.form()
    data = {str(key): form[key] for key in form}
    if "blocks" in data:
        data["blocks"] = _normalize_blocks(data.get("blocks"))
    return data


async def post_approve_to_gateway(
    *,
    settings: Settings,
    thread_id: str,
    channel: str,
    text: str,
    blocks: list[Any],
    message_ts: str,
) -> None:
    delay = max(0.0, settings.slack_emulator_approve_delay_seconds)
    if delay:
        await asyncio.sleep(delay)

    payload = build_approve_payload(
        thread_id=thread_id,
        channel=channel,
        text=text,
        blocks=blocks,
        message_ts=message_ts,
    )
    raw = urlencode({"payload": json.dumps(payload, separators=(",", ":"))}).encode(
        "utf-8"
    )
    timestamp, signature = sign_slack_request(
        signing_secret=settings.slack_signing_secret or "emulator-secret",
        body=raw,
    )
    url = settings.slack_interactivity_url
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            url,
            content=raw,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "X-Slack-Request-Timestamp": timestamp,
                "X-Slack-Signature": signature,
            },
        )
        response.raise_for_status()
    logger.info("Emulator approved thread %s via %s", thread_id, url)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Slack emulator",
        description="Accepts chat.postMessage and auto-approves via the gateway webhook.",
        version="1.0.0",
    )
    inbox: list[dict[str, Any]] = []
    app.state.inbox = inbox
    app.state.settings = settings

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/reviews")
    async def reviews() -> dict[str, Any]:
        return {"count": len(inbox), "objects": list(inbox)}

    @app.post("/api/chat.postMessage")
    async def chat_post_message(request: Request) -> JSONResponse:
        body = await _read_slack_body(request)
        blocks = _normalize_blocks(body.get("blocks"))
        text = str(body.get("text") or "")
        channel = str(body.get("channel") or settings.slack_channel or "C_EMULATOR")
        try:
            thread_id = thread_id_from_blocks(blocks)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        message_ts = f"{time.time():.6f}"
        record = {
            "ts": message_ts,
            "channel": channel,
            "text": text,
            "thread_id": thread_id,
            "blocks": blocks,
        }
        inbox.append(record)
        logger.info("Emulator accepted review card for thread %s", thread_id)

        if settings.slack_emulator_auto_approve:
            asyncio.create_task(
                _approve_with_retries(
                    settings=settings,
                    thread_id=thread_id,
                    channel=channel,
                    text=text,
                    blocks=blocks,
                    message_ts=message_ts,
                )
            )

        return JSONResponse(
            {
                "ok": True,
                "channel": channel,
                "ts": message_ts,
                "message": {
                    "ts": message_ts,
                    "text": text,
                    "blocks": blocks,
                    "type": "message",
                },
            }
        )

    return app


async def _approve_with_retries(
    *,
    settings: Settings,
    thread_id: str,
    channel: str,
    text: str,
    blocks: list[Any],
    message_ts: str,
    attempts: int = 4,
) -> None:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            await post_approve_to_gateway(
                settings=settings,
                thread_id=thread_id,
                channel=channel,
                text=text,
                blocks=blocks,
                message_ts=message_ts,
            )
            return
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Emulator webhook attempt %s/%s failed for thread %s: %s",
                attempt,
                attempts,
                thread_id,
                exc,
            )
            await asyncio.sleep(0.4 * attempt)
    logger.error(
        "Emulator could not reach the gateway for thread %s: %s",
        thread_id,
        last_error,
    )


app = create_app()
