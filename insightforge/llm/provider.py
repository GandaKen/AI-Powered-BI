"""Factory methods for LLM and embeddings providers."""

from __future__ import annotations

import logging

from langchain_ollama import ChatOllama, OllamaEmbeddings

from insightforge.config import Settings

logger = logging.getLogger(__name__)

_OPENAI_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _OPENAI_AVAILABLE = True
except ImportError:
    pass


def _openai_compat_base_url(settings: Settings) -> str:
    return f"{settings.bifrost_base_url.rstrip('/')}/v1"


def get_llm(settings: Settings, tier: str = "heavy"):
    """Return LLM client using Bifrost OpenAI-compatible API with Ollama fallback."""
    model = settings.llm_model_heavy if tier == "heavy" else settings.llm_model_light
    if _OPENAI_AVAILABLE:
        try:
            return ChatOpenAI(
                base_url=_openai_compat_base_url(settings),
                api_key="ollama",
                model=f"ollama/{model}",
                temperature=settings.llm_temperature,
            )
        except Exception:  # pragma: no cover
            logger.warning("Bifrost client failed; falling back to direct Ollama")
    return ChatOllama(model=model, temperature=settings.llm_temperature)


def get_embeddings(settings: Settings):
    """Return embeddings client using Bifrost with direct Ollama fallback."""
    if _OPENAI_AVAILABLE:
        try:
            return OpenAIEmbeddings(
                base_url=_openai_compat_base_url(settings),
                api_key="ollama",
                model=f"ollama/{settings.embedding_model}",
            )
        except Exception:  # pragma: no cover
            logger.warning("Bifrost embeddings failed; falling back to direct Ollama")
    return OllamaEmbeddings(model=settings.embedding_model)

