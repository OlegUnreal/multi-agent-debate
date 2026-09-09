"""Three agents with distinct system prompts and tool access."""
from __future__ import annotations

from .memory import AgentMemory


PROPOSER_SYS = (
    "You argue FOR the proposed decision. Be concrete, cite evidence, "
    "anticipate objections."
)
CRITIC_SYS = (
    "You argue AGAINST the proposed decision. Find risks, hidden costs, "
    "failure modes. Be specific."
)
JUDGE_SYS = (
    "You are the judge. Read both sides. Score each 0-10 on evidence quality. "
    "Reply with exactly one of: CONTINUE | ACCEPT | REJECT, then a one-line reason."
)


class Agent:
    def __init__(self, name: str, system: str):
        self.name = name
        self.system = system
        self.memory = AgentMemory()

    def speak(self, user_msg: str, llm) -> str:
        self.memory.add("user", user_msg)
        reply = llm(self.system, self.memory.as_prompt())
        # Defensive: the LLM hook must return text. If it returns a list or
        # other object (e.g. a tool-call payload), coerce to string so the
        # debate loop never crashes on a malformed response.
        if not isinstance(reply, str):
            reply = str(reply)
        self.memory.add("assistant", reply)
        return reply


def build_agents() -> tuple[Agent, Agent, Agent]:
    return Agent("proposer", PROPOSER_SYS), Agent("critic", CRITIC_SYS), Agent("judge", JUDGE_SYS)
