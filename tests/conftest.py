import pytest


@pytest.fixture
def stub_llm():
    """Deterministic stub: proposer/critic/judge cycle, ends on ACCEPT."""
    state = {"i": 0}

    def llm(system, prompt):
        state["i"] += 1
        if "Decide" in prompt or "verdict" in prompt.lower():
            return "ACCEPT, sufficient evidence."
        if "Counter" in prompt:
            return "Risks exist but manageable."
        return "Argument in favor."

    return llm


@pytest.fixture
def flaky_llm():
    """Returns empty on the first call, then a valid verdict."""
    calls = {"n": 0}

    def llm(system, prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""
        if "Decide" in prompt or "verdict" in prompt.lower():
            return "ACCEPT, good enough"
        return "for"

    return llm
