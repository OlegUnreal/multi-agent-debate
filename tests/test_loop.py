from debate.loop import run_debate


def test_terminates(stub_llm):
    rounds = run_debate("topic", stub_llm, max_rounds=3)
    assert len(rounds) <= 3
    assert rounds[-1].judge.upper().startswith(("ACCEPT", "REJECT", "CONTINUE"))
