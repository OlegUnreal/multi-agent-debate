"""OpenAI-backed speakers for the debate.

Each agent gets its own memory; the LLM is injected so tests stay offline.
"""
from __future__ import annotations

import os
from typing import Callable

from .agents import PROPOSER_SYS, CRITIC_SYS, JUDGE_SYS, Agent
from .config import SETTINGS
from .logging_config import get_logger

log = get_logger(__name__)


def _chat(system: str, prompt: str, model: str | None = None) -> str:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError

    model = model or SETTINGS.model
    client = OpenAI(api_key=SETTINGS.openai_api_key, timeout=SETTINGS.timeout)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=SETTINGS.temperature,
                timeout=SETTINGS.timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIError, RateLimitError, APITimeoutError) as exc:
            last_err = exc
            log.warning("llm_retry", extra={"attempt": attempt + 1, "error": type(exc).__name__})
            continue
    raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err


def make_llm(model: str | None = None) -> Callable[[str, str], str]:
    """Return llm(system, prompt) -> reply."""
    if not SETTINGS.has_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    m = model or SETTINGS.model
    return lambda s, p: _chat(s, p, m)


def live_debate(topic: str, model: str | None = None, max_rounds: int | None = None) -> str:
    from .loop import run_debate, verdict_of
    from .agents import build_agents

    proposer, critic, judge = build_agents()
    llm = make_llm(model)
    history = run_debate(topic, llm, max_rounds=max_rounds or SETTINGS.max_rounds)
    v = verdict_of(history)
    lines = [f"[{v}] {topic}"]
    for r in history:
        lines.append(f"\n--- Round {r.n} ---")
        lines.append(f"Proposer: {r.proposer}")
        lines.append(f"Critic: {r.critic}")
        lines.append(f"Judge: {r.judge}")
    return "\n".join(lines)
