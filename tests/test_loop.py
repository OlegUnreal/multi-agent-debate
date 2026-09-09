from debate.loop import run_debate, verdict_of, _extract_verdict


def test_terminates(stub_llm):
    rounds = run_debate("topic", stub_llm, max_rounds=3)
    assert len(rounds) <= 3
    assert rounds[-1].judge.upper().startswith(("ACCEPT", "REJECT", "CONTINUE"))


def test_extracts_verdict_from_prose():
    assert _extract_verdict("I think we should ACCEPT this plan.") == "ACCEPT"
    assert _extract_verdict("reject, too risky") == "REJECT"
    assert _extract_verdict("keep going, more evidence needed") == "CONTINUE"
    assert _extract_verdict("") == "CONTINUE"


def test_stops_on_accept():
    seq = iter(["for it", "against it", "ACCEPT, looks solid"])
    rounds = run_debate("t", lambda s, p: next(seq), max_rounds=5)
    assert len(rounds) == 1
    assert verdict_of(rounds) == "ACCEPT"


def test_stops_on_reject():
    seq = iter(["for", "against", "REJECT, dealbreaker"])
    rounds = run_debate("t", lambda s, p: next(seq), max_rounds=5)
    assert verdict_of(rounds) == "REJECT"


def test_retries_empty_response():
    calls = {"n": 0}

    def flaky(s, p):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""  # empty first try
        return "ACCEPT, good enough"

    rounds = run_debate("t", flaky, max_rounds=2)
    assert verdict_of(rounds) == "ACCEPT"
    assert calls["n"] >= 2


def test_survives_llm_exception():
    def boom(s, p):
        raise RuntimeError("boom")
    rounds = run_debate("t", boom, max_rounds=2)
    assert len(rounds) == 2
    assert "error" in rounds[0].proposer
