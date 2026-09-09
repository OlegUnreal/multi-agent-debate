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
        timeout=30,
    )
    return (resp.choices[0].message.content or "").strip()


def make_llm(model: str = "gpt-4o-mini") -> Callable[[str, str], str]:
    """Return llm(system, prompt) -> reply."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return lambda s, p: _chat(s, p, model)


def live_debate(topic: str, model: str = "gpt-4o-mini", max_rounds: int = 4) -> str:
    from .loop import run_debate, verdict_of
    from .agents import build_agents
    proposer, critic, judge = build_agents()
    llm = make_llm(model)
    history = run_debate(topic, llm, max_rounds=max_rounds)
    v = verdict_of(history)
    lines = [f"[{v}] {topic}"]
    for r in history:
        lines.append(f"\n--- Round {r.n} ---")
        lines.append(f"Proposer: {r.proposer}")
        lines.append(f"Critic: {r.critic}")
        lines.append(f"Judge: {r.judge}")
    return "\n".join(lines)
