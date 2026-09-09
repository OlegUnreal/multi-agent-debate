"""OpenAI-backed speakers for the debate.

Each agent gets its own memory; the LLM is injected so tests stay offline.
"""
from __future__ import annotations

import os
from typing import Callable

from .agents import PROPOSER_SYS, CRITIC_SYS, JUDGE_SYS, Agent


def _chat(system: str, prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return (resp.choices[0].message.content or "").strip()


def make_llm() -> Callable[[str, str], str]:
    """Return llm(system, prompt) -> reply."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return _chat


def live_debate(topic: str, model: str = "gpt-4o-mini") -> str:
    from .loop import run_debate
    from .agents import build_agents
    proposer, critic, judge = build_agents()
    llm = make_llm()
    return run_debate(topic, proposer, critic, judge, llm, max_rounds=4)
