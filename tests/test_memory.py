from debate.memory import AgentMemory


def test_isolation():
    a = AgentMemory()
    b = AgentMemory()
    a.add("user", "hi")
    assert len(b) == 0
    assert len(a) == 1
