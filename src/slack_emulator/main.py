"""Run the Slack emulator as ``python -m slack_emulator``."""

from __future__ import annotations

import logging

import uvicorn

from agent_team.settings import get_settings
from slack_emulator.app import create_app


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()
    uvicorn.run(
        create_app(settings),
        host=settings.slack_emulator_host,
        port=settings.slack_emulator_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
