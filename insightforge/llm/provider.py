"""Factory methods for LLM and embeddings providers."""

from __future__ import annotations

import logging
import time

import httpx
from langchain_ollama import ChatOllama, OllamaEmbeddings

from insightforge.config import Settings

logger = logging.getLogger(__name__)

_OPENAI_AVAILABLE = False
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    _OPENAI_AVAILABLE = True
except ImportError:
    pass

_BIFROST_REQUEST_TIMEOUT = 120
_BIFROST_PROBE_RETRY_SECS = 60


def _openai_compat_base_url(settings: Settings) -> str:
    return f"{settings.bifrost_base_url.rstrip('/')}/v1"


_bifrost_status: bool | None = None
_bifrost_last_probe: float = 0.0


def _bifrost_reachable(settings: Settings) -> bool:
    """Probe Bifrost with TTL — retries after *_BIFROST_PROBE_RETRY_SECS* on failure."""
    global _bifrost_status, _bifrost_last_probe  # noqa: PLW0603

    now = time.monotonic()
    if _bifrost_status is True:
        return True
    if _bifrost_status is False and (now - _bifrost_last_probe) < _BIFROST_PROBE_RETRY_SECS:
        return False

    _bifrost_last_probe = now
    try:
        resp = httpx.get(
            f"{_openai_compat_base_url(settings)}/models",
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                logger.info("Bifrost reachable with %d model(s)", len(data))
                _bifrost_status = True
                return True
            logger.warning("Bifrost returned an empty model list — provider not configured")
        _bifrost_status = False
        return False
    except Exception as exc:
        logger.warning("Bifrost probe failed: %s", exc)
        _bifrost_status = False
        return False


def get_llm(settings: Settings, tier: str = "heavy"):
    """Return LLM client using Bifrost OpenAI-compatible API with Ollama fallback."""
    model = settings.llm_model_heavy if tier == "heavy" else settings.llm_model_light
    if _OPENAI_AVAILABLE and _bifrost_reachable(settings):
        try:
            return ChatOpenAI(
                base_url=_openai_compat_base_url(settings),
                api_key="ollama",
                model=f"ollama/{model}",
                temperature=settings.llm_temperature,
                request_timeout=_BIFROST_REQUEST_TIMEOUT,
            )
        except Exception:
            logger.warning("Bifrost client construction failed; falling back to direct Ollama")
    return ChatOllama(
        model=model,
        temperature=settings.llm_temperature,
        request_timeout=_BIFROST_REQUEST_TIMEOUT,
    )


def get_embeddings(settings: Settings):
    """Return embeddings client using Bifrost with direct Ollama fallback."""
    if _OPENAI_AVAILABLE and _bifrost_reachable(settings):
        try:
            return OpenAIEmbeddings(
                base_url=_openai_compat_base_url(settings),
                api_key="ollama",
                model=f"ollama/{settings.embedding_model}",
                check_embedding_ctx_length=False,
                timeout=_BIFROST_REQUEST_TIMEOUT,
            )
        except Exception:
            logger.warning("Bifrost embeddings construction failed; falling back to direct Ollama")
    return OllamaEmbeddings(model=settings.embedding_model)
