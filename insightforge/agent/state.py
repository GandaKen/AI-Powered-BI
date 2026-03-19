"""State definitions for the agentic RAG graph."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """Graph state shared across all nodes."""

    query: str
    messages: list[dict[str, str]]
    safety_check: dict[str, Any]
    query_plan: dict[str, Any]
    retrieval_strategy: dict[str, Any]
    retrieved_chunks: list[str]
    assembled_context: str
    generated_response: str
    evaluation: dict[str, Any]
    final_response: str
    retry_count: int
    trace_steps: list[str]

