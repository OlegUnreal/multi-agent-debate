"""Debate loop: proposer → critic → judge, until judge says ACCEPT/REJECT."""
from __future__ import annotations

from dataclasses import dataclass

from .agents import Agent, build_agents


@dataclass
class Round:
    n: int
    proposer: str
    critic: str
    judge: str


def run_debate(topic: str, llm, max_rounds: int = 5) -> list[Round]:
    proposer, critic, judge = build_agents()
    history: list[Round] = []
    for n in range(1, max_rounds + 1):
        p = proposer.speak(f"Round {n}. Topic: {topic}. Make your case.", llm)
        c = critic.speak(f"Round {n}. Counter the proposer's argument:\n{p}", llm)
        j = judge.speak(
            f"Round {n}.\nProposer: {p}\nCritic: {c}\nDecide: CONTINUE | ACCEPT | REJECT.",
            llm,
        )
        history.append(Round(n, p, c, j))
        if j.strip().upper().startswith("ACCEPT") or j.strip().upper().startswith("REJECT"):
            break
    return history
