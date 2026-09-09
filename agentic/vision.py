"""Vision layer: turn a page snapshot into structured elements via the LLM."""
from __future__ import annotations

import json
import re
from typing import Any, Callable


_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\n?|```$", re.MULTILINE)


def _parse_elements(raw: Any) -> list[dict]:
    """Best-effort parse of an LLM element list, tolerant of fences and junk."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    text = raw if isinstance(raw, str) else str(raw)
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[: -len("```")].rstrip()
    try:
        data = json.loads(text)
    except Exception:
        return []
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    return []


def describe(state: Any, decide: Callable) -> list[dict]:
    """Ask the LLM to describe interactive elements on `state`.

    Tolerates decision-style stubs (returns []) and crashing stubs.
    """
    try:
        raw = decide(
            "list interactive elements",
            getattr(state, "url", ""),
            getattr(state, "text", ""),
            [],
        )
    except Exception:
        return []
    if isinstance(raw, dict):
        return []
    return _parse_elements(raw)
