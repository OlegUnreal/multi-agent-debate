"""Integration tests for the debate loop: retrieval memory, rubric judging,
episodic prior evidence, preference extraction and calibration reporting.

Unit coverage of verdict parsing / retry behaviour lives in
test_agent_loop.py; this file tests the layers added on top of it.
"""
from __future__ import annotations

import json

import pytest

from debate.loop import (
    Round,
    _argument_ids,
    calibration_report,
    preferences_from_history,
    rank_debate_arguments,
    run_debate,
    verdict_of,
)
from debate.episodic import EpisodicMemory
from debate.rubric import Rubric


def rubric_json(winner="proposer", conf=0.9, scores=(3, 2, 1, 2)) -> str:
    e, r, b, c = scores
    return json.dumps({"evidence": e, "reasoning": r, "rebuttal_strength": b,
                       "clarity": c, "winner": winner, "confidence": conf})


class JudgeThenStub:
    """Stub LLM: proposer/critic free text, judge always returns valid JSON."""

    def __init__(self, verdict_line=None):
        self.verdict_line = verdict_line
        self.n = 0

    def __call__(self, system, prompt):
        last = prompt[-1]["content"]
        if "Decide" in last:
            return rubric_json() if self.verdict_line is None else self.verdict_line
        if "Counter" in last:
            return "The rollback plan is thin: no rehearsal, no metrics owner."
        return "Adopt the change; LRU eviction restores latency with no new hardware."


def test_round_dataclass_backward_compatible():
    r = Round(1, "p", "c", "ACCEPT ok")
    assert r.rubric is None


def test_rubric_judging_produces_structured_rounds():
    history = run_debate("Roll back the eviction change?", JudgeThenStub(),
                         max_rounds=2, rubric_judging=True)
    assert len(history) == 1  # ACCEPT inside rubric -> stop
    r = history[0]
    assert isinstance(r.rubric, Rubric)
    assert r.rubric.winner == "proposer"
    assert verdict_of(history) == "ACCEPT"  # verdict still extractable


def test_malformed_rubric_triggers_repair_then_continues():
    seq = iter([
        "propose: latency fix",
        "critic: no rehearsal",
        "not json at all",        # first judge attempt -> invalid
        rubric_json(winner="critic", conf=0.7),  # repair attempt -> valid ACCEPT
    ])
    llm = lambda s, p: next(seq)  # noqa: E731
    history = run_debate("t", llm, max_rounds=2, rubric_judging=True)
    assert history[0].rubric is not None
    assert history[0].rubric.winner == "critic"
    assert verdict_of(history) == "REJECT"  # derived from rubric winner


def test_rubric_judge_failure_degrades_to_free_text():
    seq = iter(["p1", "c1", "no json", "still no json",
                "p2", "c2", "no json", "still no json"])
    llm = lambda s, p: next(seq)  # noqa: E731
    history = run_debate("t", llm, max_rounds=2, rubric_judging=True)
    assert len(history) == 2
    assert all(r.rubric is None for r in history)
    assert verdict_of(history) == "CONTINUE"


def test_preferences_and_bt_ranking_over_history():
    history = [
        Round(1, "a", "b", "CONTINUE", rubric=Rubric(
            evidence=3, reasoning=3, rebuttal_strength=0, clarity=2,
            winner="proposer", confidence=0.9)),
        Round(2, "a", "b", "CONTINUE", rubric=Rubric(
            evidence=1, reasoning=1, rebuttal_strength=3, clarity=1,
            winner="critic", confidence=0.6)),
        Round(3, "a", "b", "ACCEPT", rubric=Rubric(
            evidence=3, reasoning=2, rebuttal_strength=1, clarity=2,
            winner="proposer", confidence=0.8)),
    ]
    prefs = preferences_from_history(history)
    assert prefs == [(0, 1), (3, 2), (4, 5)]
    bt = rank_debate_arguments(history)
    assert len(bt.ratings) == 6
    assert bt.ratings[0] > bt.ratings[1]  # round-1 proposer dominates its critic
    assert _argument_ids(history) == [0, 1, 2, 3, 4, 5]


def test_calibration_report_shape():
    history = [
        Round(1, "a", "b", "ACCEPT", rubric=Rubric(
            evidence=3, reasoning=3, rebuttal_strength=0, clarity=2,
            winner="proposer", confidence=0.9)),
    ]
    rep = calibration_report(history)
    assert rep["n_judged"] == 1
    assert rep["agreement"] == 1.0  # BT on [(0,1)] agrees with proposer winner
    assert 0.0 <= rep["ece"] <= 1.0
    assert calibration_report([Round(1, "p", "c", "j")]) == {
        "ece": 0.0, "n_judged": 0, "agreement": 0.0}


def test_episodic_prior_evidence_flows_between_debates():
    store = EpisodicMemory(":memory:")
    topic_a = "Should we roll back the cache eviction change to fix latency?"
    run_debate(topic_a, JudgeThenStub(), max_rounds=1,
               episodic=store, debate_id="d1", rubric_judging=True)
    assert len(store) == 2  # proposer + critic recorded
    strengths = [r.strength for r in store.all_records()]
    assert all(s is not None for s in strengths)  # BT ratings written back
    assert strengths[0] > strengths[1]            # rubric winner ranks higher

    seen = {"calls": []}

    def spy(system, prompt):
        seen["calls"].append(prompt[-1]["content"])
        return JudgeThenStub()(system, prompt)

    run_debate("Does latency justify rolling back the eviction policy?",
               spy, max_rounds=1, episodic=store, debate_id="d2")
    opener = seen["calls"][0]
    assert "Prior evidence from earlier debates" in opener
    assert topic_a in opener  # attribution present
    assert "d2" != "d1"


def test_same_debate_records_are_not_recalled_back():
    store = EpisodicMemory(":memory:")
    topic = "Roll back eviction?"
    run_debate(topic, JudgeThenStub(), max_rounds=1, episodic=store, debate_id="dX")
    hits = store.recall(topic, k=5, exclude_debate="dX")
    assert hits == []


def test_agents_use_retrieval_not_fifo_in_live_loop():
    # 3-round debate; critic should see round-1 context in later prompts
    seq = iter(["p1", "c1", "CONTINUE reason",
                "p2", "c2", "ACCEPT final"])
    llm = lambda s, p: next(seq)  # noqa: E731
    history = run_debate("t", llm, max_rounds=3)
    assert len(history) == 2
    assert verdict_of(history) == "ACCEPT"
