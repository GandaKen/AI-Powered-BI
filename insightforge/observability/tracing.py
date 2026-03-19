"""Langfuse callback helper."""

from __future__ import annotations


def get_langfuse_callbacks(settings):
    """Return Langfuse callback list if configured, else empty list."""
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return []

    try:
        from langfuse.callback import CallbackHandler

        return [
            CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host or None,
            )
        ]
    except Exception:
        return []

