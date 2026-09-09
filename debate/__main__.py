"""CLI entrypoint: python -m debate "<topic>"""
from __future__ import annotations

import argparse
import sys

from .config import SETTINGS
from .logging_config import setup_logging, get_logger
from .llm import live_debate

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="debate", description="Multi-agent debate")
    parser.add_argument("topic", help="The decision or question to debate")
    parser.add_argument("--model", default=None, help="Override model (default: %(default)s)")
    parser.add_argument("--rounds", type=int, default=None, help="Max debate rounds")
    args = parser.parse_args(argv)

    if not SETTINGS.has_key:
        log.error("missing_api_key", extra={"hint": "set OPENAI_API_KEY or create a .env file"})
        return 2

    try:
        result = live_debate(args.topic, model=args.model, max_rounds=args.rounds)
    except Exception as exc:  # noqa: BLE001
        log.exception("debate_failed", extra={"error": type(exc).__name__})
        return 1

    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
