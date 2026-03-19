"""Unit tests for LLM provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from insightforge.config import Settings
from insightforge.llm.provider import get_embeddings, get_llm


def test_get_llm_uses_light_tier() -> None:
    """Light tier returns 3b model."""
    settings = Settings()
    with patch("insightforge.llm.provider.ChatOllama") as mock_ollama:
        get_llm(settings, tier="light")
        mock_ollama.assert_called_once()
        assert mock_ollama.call_args.kwargs["model"] == settings.llm_model_light


def test_get_llm_uses_heavy_tier() -> None:
    """Heavy tier returns 8b model."""
    settings = Settings()
    with patch("insightforge.llm.provider.ChatOllama") as mock_ollama:
        get_llm(settings, tier="heavy")
        mock_ollama.assert_called_once()
        assert mock_ollama.call_args.kwargs["model"] == settings.llm_model_heavy


def test_get_embeddings_returns_ollama() -> None:
    """Embeddings returns OllamaEmbeddings when OpenAI unavailable."""
    settings = Settings()
    with patch("insightforge.llm.provider.OllamaEmbeddings") as mock_emb:
        get_embeddings(settings)
        mock_emb.assert_called_once()
        assert mock_emb.call_args.kwargs["model"] == settings.embedding_model
