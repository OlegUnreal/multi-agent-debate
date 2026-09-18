"""Debate loop: proposer → critic → judge, until judge says ACCEPT/REJECT.

Extension points (all optional, defaults preserve the original behaviour):

* retrieval-augmented per-agent memory (see ``debate.memory``),
* cross-debate prior evidence via ``episodic=EpisodicMemory(...)``,
* structured rubric judging via ``rubric_judging=True``,
* batch preference extraction + Bradley–Terry ranking via
  ``preferences_from_history`` / ``rank_debate_arguments``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .agents import Agent, build_agents
from .rubric import REPAIR_INSTRUCTION, JudgeOutcome, Rubric
from .episodic import EpisodicMemory, format_prior_evidence


@dataclass
class Round:
    n: int
    proposer: str
    critic: str
    judge: str
    rubric: Optional[Rubric] = None


_VERDICT_RE = re.compile(r"\b(ACCEPT|REJECT|CONTINUE)\b", re.IGNORECASE)

JUDGE_RUBRIC_PROMPT = (
    "Round {n}.\nProposer: {p}\nCritic: {c}\n"
    "Decide with exactly one verdict: CONTINUE | ACCEPT | REJECT. "
    "Respond with ONLY a JSON object: "
    '{{"verdict": "ACCEPT|REJECT|CONTINUE", "evidence": 0-3, "reasoning": 0-3, '
    '"rebuttal_strength": 0-3, "clarity": 0-3, "winner": "proposer|critic|continue", '
    '"confidence": 0.0-1.0}}'
)


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


def _judge_with_rubric(agent: Agent, base_prompt: str, llm,
                       max_repairs: int = 1) -> JudgeOutcome:
    """Judge one round; validates the rubric and reprompts on malformed JSON."""
    from .rubric import parse_rubric, RubricParseError

    def speak(prompt: str) -> str:
        return _safe_speak(agent, prompt, llm)

    outcome = speak(base_prompt)
    raw = outcome
    try:
        return JudgeOutcome(rubric=parse_rubric(raw), raw=raw)
    except RubricParseError as exc:
        last_err = str(exc)
    attempts = 1
    for _ in range(max_repairs):
        attempts += 1
        repair_prompt = base_prompt + "\n" + REPAIR_INSTRUCTION.format(error=last_err)
        raw = speak(repair_prompt)
        try:
            return JudgeOutcome(rubric=parse_rubric(raw), raw=raw,
                                attempts=attempts, repaired=True)
        except RubricParseError as exc:
            last_err = str(exc)
    return JudgeOutcome(rubric=None, raw=raw, attempts=attempts, errors=[last_err])


def run_debate(topic: str, llm, max_rounds: int = 5, *,
               episodic: Optional[EpisodicMemory] = None,
               debate_id: str = "default",
               rubric_judging: bool = False) -> list[Round]:
    """Run one debate. Returns the history; side effects go through
    ``episodic`` (prior evidence in, arguments + BT strengths out)."""
    proposer, critic, judge = build_agents()
    history: list[Round] = []
    arg_row_ids: dict[int, int] = {}

    prior_block = ""
    if episodic is not None:
        records = episodic.recall(topic, k=3, exclude_debate=debate_id)
        prior_block = format_prior_evidence(records)

    for n in range(1, max_rounds + 1):
        p_prompt = f"Round {n}. Topic: {topic}. Make your case."
        if n == 1 and prior_block:
            p_prompt = f"{prior_block}\n\n{p_prompt}"
        p = _safe_speak(proposer, p_prompt, llm)
        c = _safe_speak(critic, f"Round {n}. Counter the proposer's argument:\n{p}", llm)
        rubric_obj = None
        if rubric_judging:
            outcome = _judge_with_rubric(
                judge,
                JUDGE_RUBRIC_PROMPT.format(n=n, p=p, c=c),
                llm,
            )
            j, rubric_obj = outcome.raw, outcome.rubric
        else:
            j = _safe_speak(
                judge,
                f"Round {n}.\nProposer: {p}\nCritic: {c}\n"
                "Decide with exactly one verdict: CONTINUE | ACCEPT | REJECT, then a one-line reason.",
                llm,
            )
        history_entry_j = j
        if rubric_obj is not None and _extract_verdict(j) == "CONTINUE" \
                and rubric_obj.winner != "continue":
            derived = "ACCEPT" if rubric_obj.winner == "proposer" else "REJECT"
            history_entry_j = f"{j}\n{derived} (verdict derived from rubric winner)"
        history.append(Round(n, p, c, history_entry_j, rubric=rubric_obj))
        if _extract_verdict(history_entry_j) in {"ACCEPT", "REJECT"}:
            break

    if episodic is not None:
        for r in history:
            arg_row_ids[2 * (r.n - 1)] = episodic.record(debate_id, topic, "proposer", r.proposer)
            arg_row_ids[2 * (r.n - 1) + 1] = episodic.record(debate_id, topic, "critic", r.critic)
        bt = rank_debate_arguments(history)
        for arg_id, rating in zip(_argument_ids(history), bt.ratings):
            if arg_id in arg_row_ids:
                episodic.set_strength(arg_row_ids[arg_id], float(rating))
    return history


# ------------------------------------------------------- ranking integration


def _argument_ids(history: list[Round]) -> list[int]:
    """Deterministic argument slot ids: proposer=2*(n-1), critic=2*(n-1)+1."""
    ids: list[int] = []
    for r in history:
        ids.extend([2 * (r.n - 1), 2 * (r.n - 1) + 1])
    return ids


def preferences_from_history(history: list[Round]) -> list[tuple[int, int]]:
    """Judge rubric winners as (winner_arg_id, loser_arg_id) pairs."""
    prefs: list[tuple[int, int]] = []
    for r in history:
        if r.rubric is None or r.rubric.winner == "continue":
            continue
        prop_id, crit_id = 2 * (r.n - 1), 2 * (r.n - 1) + 1
        if r.rubric.winner == "proposer":
            prefs.append((prop_id, crit_id))
        else:
            prefs.append((crit_id, prop_id))
    return prefs


def rank_debate_arguments(history: list[Round]):
    """Bradley–Terry fit over this debate's judge preferences.

    With no pairwise signal yet, falls back to rubric totals so callers
    always get one rating per argument slot.
    """
    from .ranking import BradleyTerryResult, fit_bradley_terry
    import numpy as np

    n_args = len(_argument_ids(history))
    prefs = preferences_from_history(history)
    if prefs:
        return fit_bradley_terry(prefs, n_items=n_args)
    ratings = np.zeros(n_args)
    for r in history:
        if r.rubric is not None:
            ratings[2 * (r.n - 1)] = float(r.rubric.total)
            ratings[2 * (r.n - 1) + 1] = float(r.rubric.total) - 0.5
    return BradleyTerryResult(ratings=ratings, converged=False)


def calibration_report(history: list[Round]) -> dict:
    """ECE + consistency between judge confidence and BT consensus.

    Binary label: was the round's rubric winner the argument that the batch
    BT model ranks higher? Deterministic given the history.
    """
    from .rubric import expected_calibration_error, judge_consistency
    from .ranking import _pairwise_probs

    rubrics = [r.rubric for r in history if r.rubric is not None]
    if not rubrics:
        return {"ece": 0.0, "n_judged": 0, "agreement": 0.0}
    bt = rank_debate_arguments(history)
    probs = _pairwise_probs(bt.ratings)
    confs: list[float] = []
    correct: list[int] = []
    for r in history:
        if r.rubric is None or r.rubric.winner == "continue":
            continue
        prop_id, crit_id = 2 * (r.n - 1), 2 * (r.n - 1) + 1
        if prop_id >= len(bt.ratings) or crit_id >= len(bt.ratings):
            continue
        bt_winner = "proposer" if probs[prop_id, crit_id] >= 0.5 else "critic"
        confs.append(r.rubric.confidence)
        correct.append(int(r.rubric.winner == bt_winner))
    return {
        "ece": expected_calibration_error(confs, correct),
        "agreement": (sum(correct) / len(correct)) if correct else 0.0,
        "n_judged": len(confs),
        **judge_consistency(rubrics),
    }


def verdict_of(history: list[Round]) -> str:
    """Return the final verdict, or CONTINUE if the debate exhausted rounds."""
    if not history:
        return "CONTINUE"
    return _extract_verdict(history[-1].judge)
