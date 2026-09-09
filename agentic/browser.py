"""Agentic browser: LLM-driven browser automation with safety guardrails."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .guard import Guard
from .vision import describe


@dataclass
class Step:
    action: str
    detail: str = ""


def _execute(driver: Any, decision: dict, guard: Guard) -> Any:
    """Run one tool call against the driver, honouring the domain guard."""
    name = (decision or {}).get("name", "")
    args = (decision or {}).get("args", {}) or {}
    if name == "goto":
        url = args.get("url", "")
        if not guard.allows(url):
            return type("S", (), {"text": "domain not allowed", "url": url})()
        return driver.goto(url)
    if name == "click":
        return driver.click(args.get("selector", ""))
    if name == "type":
        return driver.type(args.get("selector", ""), args.get("text", ""))
    if name == "done":
        return None
    return type("S", (), {"text": f"unknown action: {name}", "url": ""})()


def run(
    goal: str,
    start_url: str,
    decide: Callable,
    max_steps: int = 10,
    allowed_domains: Optional[set[str]] = None,
) -> list[Step]:
    """Drive `decide(goal, url, elements, history)` until it returns `done`.

    Every LLM exception is caught and recorded as an `error` step so a single
    bad response cannot abort the whole run.
    """
    guard = Guard(allowed_domains or {"example.com"})
    history: list[Step] = []
    url = start_url
    for _ in range(max_steps):
        try:
            els = describe(type("S", (), {"url": url, "title": "", "body": "", "text": ""})(), decide)
        except Exception:
            els = []
        try:
            decision = decide(goal, url, els, history) or {}
        except Exception as e:
            history.append(Step("error", str(e)))
            continue
        name = (decision or {}).get("name", "")
        if name == "done":
            history.append(Step("DONE", ""))
            return history
        state = _execute(type("D", (), {
            "goto": lambda u: type("S", (), {"text": "", "url": u})(),
            "click": lambda s: type("S", (), {"text": "", "url": url})(),
            "type": lambda s, t: type("S", (), {"text": "", "url": url})(),
        })(), decision, guard)
        history.append(Step(name, getattr(state, "text", "")[:200]))
        if name == "goto" and getattr(state, "url", None):
            url = state.url
    history.append(Step("DONE", "max steps"))
    return history
