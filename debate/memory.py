"""Per-agent memory: isolated conversation history."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


class AgentMemory:
    def __init__(self, max_messages: int = 40) -> None:
        self._messages: list[Message] = []
        self.max_messages = max_messages

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role, content))
        # Keep the tail so long debates don't blow the context window.
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def as_prompt(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    def clear(self) -> None:
        self._messages.clear()
