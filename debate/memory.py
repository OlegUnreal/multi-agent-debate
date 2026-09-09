"""Per-agent memory: isolated conversation history."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    role: str
    content: str


class AgentMemory:
    def __init__(self) -> None:
        self._messages: list[Message] = []

    def add(self, role: str, content: str) -> None:
        self._messages.append(Message(role, content))

    def as_prompt(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)
