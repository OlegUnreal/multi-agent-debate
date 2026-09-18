"""Three agents with distinct system prompts and tool access."""
from __future__ import annotations

from .config import SETTINGS
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
    def __init__(self, name: str, system: str,
                 memory: AgentMemory | None = None) -> None:
        self.name = name
        self.system = system
        self.memory = memory if memory is not None else AgentMemory(
            max_messages=SETTINGS.max_memory,
            token_budget=SETTINGS.memory_token_budget,
        )

    def speak(self, user_msg: str, llm, memory_query: str | None = "") -> str:
        """Say one turn. ``memory_query`` selects the prompt-assembly path:

        - ``""`` (default): retrieve relevant past turns for ``user_msg``;
        - ``None``: blind FIFO fallback (original behaviour);
        - any other string: recall for that explicit query.
        """
        query = user_msg if memory_query == "" else memory_query
        self.memory.add("user", user_msg)
        prompt = (self.memory.as_prompt(query=query) if query
                  else self.memory.as_prompt())
        reply = llm(self.system, prompt)
        # Defensive: the LLM hook must return text. If it returns a list or
        # other object (e.g. a tool-call payload), coerce to string so the
        # debate loop never crashes on a malformed response.
        if not isinstance(reply, str):
            reply = str(reply)
        self.memory.add("assistant", reply)
        return reply


def build_agents() -> tuple[Agent, Agent, Agent]:
    return Agent("proposer", PROPOSER_SYS), Agent("critic", CRITIC_SYS), Agent("judge", JUDGE_SYS)
