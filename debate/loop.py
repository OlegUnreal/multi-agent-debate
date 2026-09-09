"""Debate loop: proposer → critic → judge, until judge says ACCEPT/REJECT."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .agents import Agent, build_agents


@dataclass
class Round:
    n: int
    proposer: str
    critic: str
    judge: str


_VERDICT_RE = re.compile(r"\b(ACCEPT|REJECT|CONTINUE)\b", re.IGNORECASE)


def _as_text(value) -> str:
    """Coerce any LLM payload to a single string.

    Models sometimes return a list/dict (e.g. a tool-call payload) instead of
    text. Joining list items keeps the content readable instead of producing
    the useless repr "['x']".
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _extract_verdict(text: str) -> str:
    """Pull the first verdict token out of free-form judge text."""
    m = _VERDICT_RE.search(text or "")
    return m.group(1).upper() if m else "CONTINUE"


def _safe_speak(agent: Agent, prompt: str, llm, retries: int = 2) -> str:
    """Call the agent, retrying on empty/exception so one bad round can't kill the debate."""
    last = ""
    for attempt in range(retries + 1):
        try:
            out = _as_text(agent.speak(prompt, llm))
        except Exception as exc:  # noqa: BLE001
            last = f"[error: {exc}]"
            continue
        if out and out.strip():
            return out
        last = out or ""
    return last or "[no response]"


def run_debate(topic: str, llm, max_rounds: int = 5) -> list[Round]:
    proposer, critic, judge = build_agents()
    history: list[Round] = []
    for n in range(1, max_rounds + 1):
        p = _safe_speak(proposer, f"Round {n}. Topic: {topic}. Make your case.", llm)
        c = _safe_speak(critic, f"Round {n}. Counter the proposer's argument:\n{p}", llm)
        j = _safe_speak(
            judge,
            f"Round {n}.\nProposer: {p}\nCritic: {c}\n"
            "Decide with exactly one verdict: CONTINUE | ACCEPT | REJECT, then a one-line reason.",
            llm,
        )
        history.append(Round(n, p, c, j))
        if _extract_verdict(j) in {"ACCEPT", "REJECT"}:
            break
    return history


def verdict_of(history: list[Round]) -> str:
    """Return the final verdict, or CONTINUE if the debate exhausted rounds."""
    if not history:
        return "CONTINUE"
    return _extract_verdict(history[-1].judge)
