from debate.memory import AgentMemory


def test_truncates_to_max():
    m = AgentMemory(max_messages=4)
    for i in range(10):
        m.add("user", f"msg {i}")
    assert len(m) == 4
    assert m._messages[0].content == "msg 6"


def test_clear():
    m = AgentMemory()
    m.add("user", "x")
    m.clear()
    assert len(m) == 0
