"""EXPERIMENTAL browser-agent prototype.

This package (guard / vision / browser) is a self-contained LLM-driven
browser loop kept for design exploration. It is **not** part of the debate
pipeline: nothing in ``debate/`` imports it, and it is not wired into the
server or CLI. Its behaviour is pinned by ``tests/test_agentic.py`` so any
future change to these semantics is a deliberate one.
"""
