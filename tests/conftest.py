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
