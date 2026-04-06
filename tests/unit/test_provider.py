"""Unit tests for LLM provider."""

from __future__ import annotations

from unittest.mock import patch

from insightforge.config import Settings
from insightforge.llm.provider import get_embeddings, get_llm


def test_get_llm_uses_light_tier() -> None:
    """Light tier returns correct model via Bifrost (ChatOpenAI) or Ollama fallback."""
    settings = Settings()
    with patch("insightforge.llm.provider.ChatOpenAI") as mock_openai, \
         patch("insightforge.llm.provider._OPENAI_AVAILABLE", True):
        get_llm(settings, tier="light")
        mock_openai.assert_called_once()
        assert settings.llm_model_light in mock_openai.call_args.kwargs["model"]


def test_get_llm_uses_heavy_tier() -> None:
    """Heavy tier returns correct model via Bifrost (ChatOpenAI) or Ollama fallback."""
    settings = Settings()
    with patch("insightforge.llm.provider.ChatOpenAI") as mock_openai, \
         patch("insightforge.llm.provider._OPENAI_AVAILABLE", True):
        get_llm(settings, tier="heavy")
        mock_openai.assert_called_once()
        assert settings.llm_model_heavy in mock_openai.call_args.kwargs["model"]


def test_get_llm_falls_back_to_ollama() -> None:
    """When OpenAI is unavailable, falls back to direct ChatOllama."""
    settings = Settings()
    with patch("insightforge.llm.provider._OPENAI_AVAILABLE", False), \
         patch("insightforge.llm.provider.ChatOllama") as mock_ollama:
        get_llm(settings, tier="light")
        mock_ollama.assert_called_once()
        assert mock_ollama.call_args.kwargs["model"] == settings.llm_model_light


def test_get_embeddings_returns_openai_compat() -> None:
    """Embeddings use Bifrost OpenAI-compat endpoint when available."""
    settings = Settings()
    with patch("insightforge.llm.provider.OpenAIEmbeddings") as mock_emb, \
         patch("insightforge.llm.provider._OPENAI_AVAILABLE", True):
        get_embeddings(settings)
        mock_emb.assert_called_once()
        assert settings.embedding_model in mock_emb.call_args.kwargs["model"]


def test_get_embeddings_falls_back_to_ollama() -> None:
    """When OpenAI is unavailable, falls back to direct OllamaEmbeddings."""
    settings = Settings()
    with patch("insightforge.llm.provider._OPENAI_AVAILABLE", False), \
         patch("insightforge.llm.provider.OllamaEmbeddings") as mock_emb:
        get_embeddings(settings)
        mock_emb.assert_called_once()
        assert mock_emb.call_args.kwargs["model"] == settings.embedding_model
