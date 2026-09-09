from debate.memory import AgentMemory


def test_isolation():
    a = AgentMemory()
    b = AgentMemory()
    a.add("user", "hi")
    assert len(b) == 0
    assert len(a) == 1


def test_truncation():
    m = AgentMemory(max_messages=4)
    for i in range(10):
        m.add("user", f"msg {i}")
    assert len(m) == 4
    assert m.as_prompt()[-1]["content"] == "msg 9"


def test_clear():
    m = AgentMemory()
    m.add("user", "x")
    m.clear()
    assert len(m) == 0
