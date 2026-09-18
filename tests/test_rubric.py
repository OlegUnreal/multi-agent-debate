"""Rubric validation, malformed-output repair paths, and ECE on a
hand-checkable case."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from debate.rubric import (
    Rubric,
    RubricParseError,
    expected_calibration_error,
    judge_consistency,
    judge_round,
    parse_rubric,
)


VALID = ('{"evidence": 3, "reasoning": 2, "rebuttal_strength": 1, '
         '"clarity": 2, "winner": "proposer", "confidence": 0.8}')


def test_strict_validation_rejects_out_of_range_and_bools():
    r = parse_rubric(VALID)
    assert r.total == 8 and r.winner == "proposer"
    with pytest.raises(ValidationError):
        Rubric(evidence=4, reasoning=0, rebuttal_strength=0, clarity=0,
               winner="proposer", confidence=0.5)
    with pytest.raises(ValidationError):
        Rubric(evidence=True, reasoning=0, rebuttal_strength=0, clarity=0,
               winner="proposer", confidence=0.5)
    with pytest.raises(ValidationError):
        Rubric(evidence=1, reasoning=1, rebuttal_strength=1, clarity=1,
               winner="nobody", confidence=0.5)
    with pytest.raises(ValidationError):
        Rubric(evidence=1, reasoning=1, rebuttal_strength=1, clarity=1,
               winner="proposer", confidence=1.5)


def test_free_repair_salvages_fenced_and_partial_json():
    fenced = parse_rubric("```json\n" + VALID + "\n```")
    assert fenced.confidence == 0.8
    noisy = parse_rubric("My verdict: {" + '"evidence": "3", "reasoning": 9, '
                         '"rebuttal_strength": -2, "clarity": 2.6, '
                         '"winner": "PROPOSER", "confidence": 2}')
    assert noisy.evidence == 3
    assert noisy.reasoning == 3          # clamped
    assert noisy.rebuttal_strength == 0  # clamped
    assert noisy.clarity == 3            # rounded from 2.6
    assert noisy.winner == "proposer"
    assert noisy.confidence == 1.0       # clamped


def test_unsalvageable_text_raises_parse_error():
    for junk in ["ACCEPT, proposer wins", "", "{{{", '{"winner": "x"']:
        with pytest.raises(RubricParseError):
            parse_rubric(junk)


def test_prompted_repair_used_when_first_output_is_garbage():
    calls = {"n": 0}

    def speak(prompt):
        calls["n"] += 1
        return "ACCEPT, no JSON here" if calls["n"] == 1 else VALID

    out = judge_round(speak, "p-arg", "c-arg", lambda ctx: "Decide. " + (ctx or ""),
                      max_repairs=2)
    assert calls["n"] == 2
    assert out.repaired and out.attempts == 2
    assert out.rubric is not None and out.rubric.winner == "proposer"


def test_judge_round_gives_up_cleanly_after_max_repairs():
    out = judge_round(lambda prompt: "no json", "p", "c",
                      lambda ctx: "Decide.", max_repairs=1)
    assert out.rubric is None
    assert out.attempts == 2
    assert out.errors and "no JSON object" in out.errors[0]


# ---------------------------------------------------------------- ECE


def test_ece_hand_checkable_case():
    # two correct @0.9, two wrong @0.6, 10 equal-width bins:
    # bin [0.8,0.9): |1.0-0.9| weight 2/4 ; bin [0.6,0.7): |0-0.6| weight 2/4
    # ECE = 0.5*0.1 + 0.5*0.6 = 0.35
    ece = expected_calibration_error([0.9, 0.9, 0.6, 0.6], [1, 1, 0, 0], n_bins=10)
    assert ece == pytest.approx(0.35, abs=1e-9)


def test_ece_perfect_calibration_is_zero():
    confs = [1.0, 0.5, 0.5, 1.0, 0.0, 0.0]
    correct = [1, 1, 0, 1, 0, 0]  # bin avg conf == bin accuracy everywhere
    assert expected_calibration_error(confs, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_empty_and_mismatched():
    assert expected_calibration_error([], []) == 0.0
    with pytest.raises(ValueError):
        expected_calibration_error([0.5], [1, 0])


def test_judge_consistency_metrics():
    mk = lambda w: Rubric(evidence=2, reasoning=2, rebuttal_strength=2,
                          clarity=2, winner=w, confidence=0.7)
    c = judge_consistency([mk("proposer"), mk("proposer"), mk("critic")])
    assert c["winner_stability"] == pytest.approx(0.5)
    assert c["score_spread"] == 0.0
    assert judge_consistency([]) == {"winner_stability": 0.0,
                                     "score_spread": 0.0, "n": 0}
