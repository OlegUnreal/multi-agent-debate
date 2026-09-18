"""Central configuration loaded from environment / .env.

Stdlib `dataclass(frozen=True)` + `field(default_factory=...)` — deliberately
not pydantic-settings: a handful of scalars does not warrant a validation
framework, and the defaults are read from os.environ at import time.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: os.environ.get("DEBATE_MODEL", "gpt-4o-mini"))
    max_rounds: int = field(default_factory=lambda: int(os.environ.get("DEBATE_MAX_ROUNDS", "4")))
    temperature: float = field(default_factory=lambda: float(os.environ.get("DEBATE_TEMPERATURE", "0.4")))
    timeout: int = field(default_factory=lambda: int(os.environ.get("DEBATE_TIMEOUT", "30")))
    max_memory: int = field(default_factory=lambda: int(os.environ.get("DEBATE_MAX_MEMORY", "40")))
    memory_token_budget: int = field(
        default_factory=lambda: int(os.environ.get("DEBATE_MEMORY_TOKEN_BUDGET", "900")))
    episodic_db: str = field(
        default_factory=lambda: os.environ.get("DEBATE_EPISODIC_DB", ":memory:"))
    host: str = field(default_factory=lambda: os.environ.get("DEBATE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("DEBATE_PORT", "8000")))

    @property
    def has_key(self) -> bool:
        return bool(self.openai_api_key)


SETTINGS = Settings()
