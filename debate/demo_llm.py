"""Live debate with a real LLM.

Usage: OPENAI_API_KEY=... python -m debate.demo_llm "Should we ship on Friday?"
"""
from __future__ import annotations

import sys

from .llm import live_debate


def main() -> None:
    topic = " ".join(sys.argv[1:]) or "Should we migrate the monolith to microservices?"
    print(live_debate(topic))


if __name__ == "__main__":
    main()
