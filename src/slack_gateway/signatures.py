"""Slack request signature verification."""

from __future__ import annotations

import hashlib
import hmac
import time


def sign_slack_request(
    *,
    signing_secret: str,
    body: bytes,
    timestamp: str | None = None,
) -> tuple[str, str]:
    """Return ``(timestamp, v0=hex)`` for a Slack-compatible request signature."""
    ts = timestamp or str(int(time.time()))
    basestring = b"v0:" + ts.encode() + b":" + body
    digest = hmac.new(signing_secret.encode(), basestring, hashlib.sha256).hexdigest()
    return ts, f"v0={digest}"


def verify_slack_signature(
    *,
    signing_secret: str,
    body: bytes,
    timestamp: str,
    signature: str,
    max_age_seconds: int = 300,
) -> bool:
    if not signing_secret or not timestamp or not signature:
        return False
    try:
        if abs(time.time() - int(timestamp)) > max_age_seconds:
            return False
    except ValueError:
        return False

    _ts, expected = sign_slack_request(
        signing_secret=signing_secret, body=body, timestamp=timestamp
    )
    return hmac.compare_digest(expected, signature)
