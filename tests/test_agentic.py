"""Behavior-lock tests for the EXPERIMENTAL `agentic/` package.

Nothing in `debate/` imports this package (verified: the only reference to
"agentic" in the codebase is the package itself), so rather than wire it in
or delete it, these tests pin the current semantics of the guard, the vision
parser, and the browser loop. Any change to a locked behavior must be
deliberate.
"""
from __future__ import annotations

from agentic.browser import Step, run
from agentic.guard import Guard
from agentic.vision import _parse_elements, describe


# -- guard -------------------------------------------------------------------

def test_guard_allows_exact_domain_and_subdomains():
    g = Guard({"Example.com"})                       # case normalised
    assert g.allows("https://example.com/a/b?x=1")
    assert g.allows("https://login.example.com")
    assert not g.allows("https://notexample.com")    # no suffix trickery
    assert not g.allows("https://example.com.evil.io/x")
    assert not g.allows("")
    assert not g.allows("not a url")


def test_guard_empty_allowlist_denies_everything():
    g = Guard(set())
    assert not g.allows("https://example.com")


# -- vision --------------------------------------------------------------------

def test_parse_elements_accepts_dicts_and_fenced_json():
    assert _parse_elements(None) == []
    assert _parse_elements([{"a": 1}, "junk", 7]) == [{"a": 1}]
    assert _parse_elements('```json\n[{"id": 1}]\n```') == [{"id": 1}]
    assert _parse_elements('[{"id": 2}]') == [{"id": 2}]


def test_parse_elements_rejects_junk_and_non_lists():
    assert _parse_elements("sure, here you go") == []
    assert _parse_elements('{"a": 1}') == []          # object, not element list


def test_describe_calls_decide_and_tolerates_failures():
    state = type("S", (), {"url": "https://example.com", "text": "hello"})()
    els = describe(state, lambda *a: '[{"role": "button"}]')
    assert els == [{"role": "button"}]
    assert describe(state, lambda *a: {"name": "done"}) == []   # decision-like

    def boom(*a):
        raise RuntimeError("llm down")
    assert describe(state, boom) == []


# -- browser loop --------------------------------------------------------------

class Decider:
    """Scripted decide(): answers element queries with [], replays decisions.

    An Exception entry raises, mimicking a flaky LLM mid-run.
    """

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.n = 0

    def __call__(self, prompt, url="", extra=None, history=None):
        if prompt == "list interactive elements":
            return []
        d = self.decisions[self.n]
        self.n += 1
        if isinstance(d, Exception):
            raise d
        return d


def test_run_returns_done_immediately():
    hist = run("goal", "https://example.com", Decider([{"name": "done"}]))
    assert hist == [Step("DONE", "")]


def test_run_goto_allowed_then_done():
    hist = run("goal", "https://example.com/start", Decider([
        {"name": "goto", "args": {"url": "https://example.com/next"}},
        {"name": "done"},
    ]))
    assert [s.action for s in hist] == ["goto", "DONE"]


def test_run_blocked_domain_is_stepped_not_navigated():
    hist = run("goal", "https://example.com", Decider([
        {"name": "goto", "args": {"url": "https://evil.test/x"}},
        {"name": "done"},
    ]))
    assert hist[0] == Step("goto", "domain not allowed")
    assert [s.action for s in hist] == ["goto", "DONE"]


def test_run_custom_allowlist_overrides_default():
    hist = run("goal", "https://example.com", Decider([
        {"name": "goto", "args": {"url": "https://trusted.dev/app"}},
        {"name": "done"},
    ]), allowed_domains={"trusted.dev"})
    assert hist == [Step("goto", ""), Step("DONE", "")]
    hist2 = run("goal", "https://example.com", Decider([
        {"name": "goto", "args": {"url": "https://example.com/legacy"}},
        {"name": "done"},
    ]), allowed_domains={"trusted.dev"})
    assert hist2[0].detail == "domain not allowed"     # default no longer applies


def test_run_click_and_type_pass_through():
    hist = run("goal", "https://example.com", Decider([
        {"name": "click", "args": {"selector": "#go"}},
        {"name": "type", "args": {"selector": "#q", "text": "hi"}},
        {"name": "done"},
    ]))
    assert [s.action for s in hist] == ["click", "type", "DONE"]


def test_run_unknown_action_recorded():
    hist = run("goal", "https://example.com", Decider([
        {"name": "wave"},
        {"name": "done"},
    ]))
    assert hist[0] == Step("wave", "unknown action: wave")


def test_run_llm_exception_becomes_error_step_and_continues():
    hist = run("goal", "https://example.com", Decider([
        RuntimeError("boom"),
        {"name": "done"},
    ]))
    assert hist == [Step("error", "boom"), Step("DONE", "")]


def test_run_max_steps_terminates_loop():
    hist = run("goal", "https://example.com",
               Decider([{"name": "wave"}, {"name": "wave"}]), max_steps=2)
    assert [s.action for s in hist] == ["wave", "wave", "DONE"]
    assert hist[-1].detail == "max steps"
