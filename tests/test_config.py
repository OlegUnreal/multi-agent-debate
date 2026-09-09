import os

from debate import config


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("DEBATE_MODEL", "gpt-4o")
    monkeypatch.setenv("DEBATE_MAX_ROUNDS", "6")
    s = config.Settings()
    assert s.openai_api_key == "sk-test"
    assert s.model == "gpt-4o"
    assert s.max_rounds == 6
    assert s.has_key is True


def test_settings_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = config.Settings()
    assert s.has_key is False
