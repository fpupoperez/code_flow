"""JetStream helpers shared by publishers and queue-group subscribers."""

from __future__ import annotations

import asyncio
import logging
import signal
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import nats
from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.js import JetStreamContext
from nats.js.api import (
    AckPolicy,
    ConsumerConfig,
    DeliverPolicy,
    RetentionPolicy,
    StorageType,
    StreamConfig,
)
from nats.js.errors import NotFoundError
from opentelemetry.trace import SpanKind
from pydantic import BaseModel

from telemetry.otel import start_span

logger = logging.getLogger(__name__)

MessageHandler = Callable[[Msg], Awaitable[None]]
EventModel = TypeVar("EventModel", bound=BaseModel)


def _msg_header(msg: Msg, key: str) -> str | None:
    headers = msg.headers
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if getter is None:
        return None
    value = getter(key)
    return str(value) if value is not None else None


def covering_subjects(subject: str) -> list[str]:
    """Widen ``a.b.c`` to ``a.b.>`` so one stream holds related topics."""
    parts = subject.split(".")
    if len(parts) >= 2:
        return [".".join(parts[:-1]) + ".>"]
    return [subject, f"{subject}.>"]


async def connect(url: str, *, name: str = "agent-team") -> Client:
    return await nats.connect(
        servers=[url],
        name=name,
        connect_timeout=5,
        max_reconnect_attempts=8,
        reconnect_time_wait=1,
    )


async def ensure_stream(
    js: JetStreamContext,
    *,
    stream: str,
    subject: str,
) -> None:
    subjects = covering_subjects(subject)
    try:
        info = await js.stream_info(stream)
    except NotFoundError:
        await js.add_stream(
            StreamConfig(
                name=stream,
                subjects=subjects,
                retention=RetentionPolicy.LIMITS,
                storage=StorageType.FILE,
                max_age=7 * 24 * 60 * 60,
                duplicate_window=120,
                num_replicas=1,
            )
        )
        logger.info("Created JetStream stream %s for %s", stream, subjects)
        return

    current = list(info.config.subjects or [])
    merged = list(dict.fromkeys(current + subjects))
    if merged != current:
        await js.update_stream(name=stream, subjects=merged)
        logger.info("Updated JetStream stream %s subjects to %s", stream, merged)


async def publish_event(
    js: JetStreamContext,
    *,
    subject: str,
    event: BaseModel,
) -> None:
    event_id = str(getattr(event, "event_id", "") or "")
    headers = {"Nats-Msg-Id": event_id} if event_id else None
    attributes = {
        "messaging.system": "nats",
        "messaging.operation": "publish",
        "messaging.destination.name": subject,
        "messaging.message.id": event_id or None,
        "messaging.nats.event_type": getattr(event, "event_type", None)
        or event.__class__.__name__,
    }
    with start_span("nats.publish", kind=SpanKind.PRODUCER, attributes=attributes):
        ack = await js.publish(
            subject,
            event.model_dump_json().encode("utf-8"),
            headers=headers,
        )
    logger.info(
        "Published %s to %s (stream=%s seq=%s)",
        event_id or event.__class__.__name__,
        subject,
        ack.stream,
        ack.seq,
    )


@asynccontextmanager
async def jetstream_connection(
    url: str, *, name: str = "agent-team"
) -> AsyncIterator[tuple[Client, JetStreamContext]]:
    nc = await connect(url, name=name)
    try:
        yield nc, nc.jetstream()
    finally:
        await nc.drain()


async def push_subscribe(
    js: JetStreamContext,
    *,
    stream: str,
    subject: str,
    queue: str,
    cb: MessageHandler,
):
    """Bind a worker-queue push consumer.

    JetStream load-balances each message to one member of ``queue``. In nats-py
    the queue name is also the durable consumer name and deliver group, so
    every instance must subscribe with the same ``queue`` value.
    """
    await ensure_stream(js, stream=stream, subject=subject)
    return await js.subscribe(
        subject,
        queue=queue,
        cb=cb,
        durable=queue,
        stream=stream,
        manual_ack=True,
        idle_heartbeat=5.0,
        config=ConsumerConfig(
            durable_name=queue,
            deliver_group=queue,
            ack_policy=AckPolicy.EXPLICIT,
            deliver_policy=DeliverPolicy.ALL,
            ack_wait=30,
            max_deliver=8,
            max_ack_pending=16,
            filter_subject=subject,
            idle_heartbeat=5.0,
        ),
    )


async def run_queue_worker(
    *,
    nats_url: str,
    client_name: str,
    stream: str,
    subject: str,
    queue: str,
    handler: MessageHandler,
) -> None:
    """Connect, bind a push queue group, and block until SIGINT/SIGTERM."""
    nc = await connect(nats_url, name=client_name)
    js = nc.jetstream()

    async def _on_message(msg: Msg) -> None:
        attributes = {
            "messaging.system": "nats",
            "messaging.operation": "receive",
            "messaging.destination.name": subject,
            "messaging.nats.stream": stream,
            "messaging.nats.queue": queue,
            "messaging.message.id": _msg_header(msg, "Nats-Msg-Id"),
        }
        with start_span("nats.consume", kind=SpanKind.CONSUMER, attributes=attributes):
            await handler(msg)

    await push_subscribe(
        js,
        stream=stream,
        subject=subject,
        queue=queue,
        cb=_on_message,
    )
    logger.info(
        "%s listening on %s / %s (queue=%s)",
        client_name,
        stream,
        subject,
        queue,
    )

    stop = asyncio.Event()

    def _request_stop() -> None:
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            break

    try:
        await stop.wait()
    finally:
        await nc.drain()


def decode_event(msg: Msg, model: type[EventModel]) -> EventModel:
    return model.model_validate_json(msg.data)
