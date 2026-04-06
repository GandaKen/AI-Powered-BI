from __future__ import annotations

import pytest

from insightforge.config import Settings


def test_settings_defaults() -> None:
    cfg = Settings()
    assert cfg.bifrost_base_url.startswith("http")
    assert cfg.rag_top_k > 0
    assert cfg.max_eval_retries >= 0


def test_settings_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "10")
    monkeypatch.setenv("MAX_EVAL_RETRIES", "2")
    cfg = Settings()
    assert cfg.rag_top_k == 10
    assert cfg.max_eval_retries == 2


def test_settings_reject_empty_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_LIGHT", "")
    with pytest.raises(ValueError, match="cannot be empty"):
        Settings()


def test_settings_validate_url(monkeypatch) -> None:
    monkeypatch.setenv("BIFROST_BASE_URL", "invalid")
    with pytest.raises(ValueError, match="http"):
        Settings()
