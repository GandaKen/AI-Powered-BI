"""Observability callback helpers — dual-write to Langfuse Cloud + local Postgres."""

from __future__ import annotations

from insightforge.observability.collector import TraceCollector


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


def make_trace_collector(settings) -> TraceCollector | None:
    """Create a per-request TraceCollector if a database URL is configured."""
    db_url = getattr(settings, "database_url", "")
    if not db_url:
        return None
    return TraceCollector(
        database_url=db_url,
        model_name=settings.llm_model_heavy,
    )


def get_all_callbacks(settings):
    """Return combined callback list: Langfuse (cloud) + local TraceCollector.

    Returns (callbacks_list, collector_or_None) so the caller can finalize
    the collector after invocation.
    """
    callbacks = get_langfuse_callbacks(settings)
    collector = make_trace_collector(settings)
    if collector is not None:
        callbacks.append(collector)
    return callbacks, collector
