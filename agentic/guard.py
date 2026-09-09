"""Safety guard: only allow navigation to approved domains."""
from __future__ import annotations

from urllib.parse import urlparse


class Guard:
    def __init__(self, allowed: set[str]):
        self.allowed = {d.lower() for d in allowed}

    def allows(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return any(host == d or host.endswith("." + d) for d in self.allowed)
