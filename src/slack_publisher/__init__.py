"""Microservice that consumes NATS review events and posts them to Slack."""

from slack_publisher.main import main

__all__ = ["main"]
