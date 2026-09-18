"""Structured judging: Pydantic rubric, malformed-output repair, calibration.

The judge is asked for a small JSON rubric (evidence / reasoning / rebuttal /
clarity, 0-3 each, plus winner and confidence). ``parse_rubric`` validates
strictly; ``judge_round`` adds two repair layers before giving up:

1. *free* repair — extract JSON from fences/parentheses, coerce types, clamp
   ordinal fields, derive confidence defaults (no extra LLM call);
2. *prompted* repair — one retry asking the model to fix its answer, with the
   validation error echoed back (bounded by ``max_repairs``).

``expected_calibration_error`` measures judge confidence against the
ground-truth-ish signal of the Bradley–Terry consensus, and
``judge_consistency`` measures round-to-round stability.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

RUBRIC_FIELDS = ("evidence", "reasoning", "rebuttal_strength", "clarity")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class Rubric(BaseModel):
    """One judged round. Ordinals are strict ints in 0..3."""

    evidence: int = Field(ge=0, le=3, description="Proposer evidence quality")
    reasoning: int = Field(ge=0, le=3, description="Reasoning chain quality")
    rebuttal_strength: int = Field(ge=0, le=3, description="How well the critic rebutted")
    clarity: int = Field(ge=0, le=3, description="Presentation clarity, both sides")
    winner: Literal["proposer", "critic", "continue"]
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("evidence", "reasoning", "rebuttal_strength", "clarity", mode="before")
    @classmethod
    def _reject_bools_and_strings(cls, v):
        # bool is a subclass of int in Python; "2" is accepted then validated.
        if isinstance(v, bool):
            raise ValueError("rubric scores must be integers, not bool")
        if isinstance(v, str):
            return int(v.strip())
        return v

    @property
    def total(self) -> int:
        return sum(getattr(self, f) for f in RUBRIC_FIELDS)


class RubricParseError(ValueError):
    """Raised when judge output cannot be turned into a valid rubric."""


def _free_repair(raw: str) -> dict | None:
    """Best-effort salvage of a JSON object from messy model text."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    def clamp_int(value, default=0):
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            return default
        return max(0, min(3, v))

    out = {f: clamp_int(data.get(f)) for f in RUBRIC_FIELDS}
    winner = str(data.get("winner", "continue")).lower().strip()
    out["winner"] = winner if winner in {"proposer", "critic", "continue"} else "continue"
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    out["confidence"] = max(0.0, min(1.0, conf))
    return out


def parse_rubric(text: str) -> Rubric:
    """Strict parse with free repair; raises RubricParseError on garbage."""
    repaired = _free_repair(text or "")
    if repaired is None:
        raise RubricParseError("no JSON object found in judge output")
    try:
        return Rubric(**repaired)
    except ValidationError as exc:
        raise RubricParseError(str(exc)) from exc


REPAIR_INSTRUCTION = (
    "Your previous reply was not a valid rubric ({error}). Reply again with "
    "ONLY a JSON object with keys evidence, reasoning, rebuttal_strength, "
    "clarity (integers 0-3), winner ('proposer'|'critic'|'continue') and "
    "confidence (float 0-1)."
)


@dataclass
class JudgeOutcome:
    rubric: Rubric | None
    raw: str
    attempts: int = 1
    repaired: bool = False
    errors: list[str] = field(default_factory=list)


def judge_round(speak, proposer: str, critic: str, prompt_fn,
                max_repairs: int = 1) -> JudgeOutcome:
    """Ask the judge (via ``speak(prompt) -> str``) and validate the rubric.

    ``prompt_fn(round_ctx)`` builds the prompt; on invalid output we retry up
    to ``max_repairs`` times with REPAIR_INSTRUCTION appended. Verdict text is
    preserved even if every parse fails (the loop still needs a verdict).
    """
    raw = speak(prompt_fn(None))
    try:
        return JudgeOutcome(rubric=parse_rubric(raw), raw=raw)
    except RubricParseError as exc:
        last_err = str(exc)
    attempts = 1
    for _ in range(max_repairs):
        attempts += 1
        raw = speak(prompt_fn(last_err))
        try:
            rubric = parse_rubric(raw)
            return JudgeOutcome(rubric=rubric, raw=raw, attempts=attempts, repaired=True)
        except RubricParseError as exc:
            last_err = str(exc)
    return JudgeOutcome(rubric=None, raw=raw, attempts=attempts, errors=[last_err])


# ---------------------------------------------------------------- calibration


def expected_calibration_error(confidences: list[float], correct: list[int],
                               n_bins: int = 10) -> float:
    """Equal-width ECE: weighted mean of |accuracy - confidence| per bin."""
    if not confidences:
        return 0.0
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have the same length")
    total = len(confidences)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(int(correct[i]) for i in idx) / len(idx)
        conf = sum(float(confidences[i]) for i in idx) / len(idx)
        ece += (len(idx) / total) * abs(acc - conf)
    return ece


def judge_consistency(rubrics: list[Rubric]) -> dict[str, float]:
    """Round-to-round stability of the judge.

    winner_stability: fraction of consecutive rounds agreeing on the winner.
    score_spread: population std of rubric totals (lower = steadier judge).
    """
    if not rubrics:
        return {"winner_stability": 0.0, "score_spread": 0.0, "n": 0}
    winners = [r.winner for r in rubrics]
    agree = sum(1 for a, b in zip(winners, winners[1:]) if a == b)
    stability = agree / (len(winners) - 1) if len(winners) > 1 else 1.0
    totals = [r.total for r in rubrics]
    import statistics
    spread = statistics.pstdev(totals) if len(totals) > 1 else 0.0
    return {"winner_stability": stability, "score_spread": spread, "n": len(rubrics)}
