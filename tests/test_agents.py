from debate.agents import build_agents, Agent


def test_build_agents_returns_three():
    p, c, j = build_agents()
    assert isinstance(p, Agent) and p.name == "proposer"
    assert isinstance(c, Agent) and c.name == "critic"
    assert isinstance(j, Agent) and j.name == "judge"
    assert p.memory is not c.memory  # isolated memories


def test_agent_speak_records_both_sides():
    a = Agent("t", "sys")
    seen = {}

    def fake_llm(system, prompt):
        seen["system"] = system
        seen["prompt"] = prompt
        return "reply"

    out = a.speak("hello", fake_llm)
    assert out == "reply"
    assert len(a.memory) == 2
    assert a.memory._messages[0].role == "user"
    assert a.memory._messages[1].role == "assistant"
