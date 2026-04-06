"""Tool execution node."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def make_information_retriever_node(tool_registry: dict):
    """Create information retriever node."""

    def _node(state: dict) -> dict:
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append("InformationRetriever")

        strategy = state.get("retrieval_strategy", {})
        if strategy.get("blocked"):
            return {"retrieved_chunks": [], "trace_steps": trace_steps}

        chunks: list[str] = []
        failures = 0
        calls = strategy.get("calls", [])

        for call in calls:
            tool_name = call.get("tool")
            args = call.get("args", {})
            tool = tool_registry.get(tool_name)
            if tool is None:
                failures += 1
                continue
            try:
                chunks.append(str(tool.invoke(args)))
            except Exception:
                failures += 1
                logger.exception("Tool '%s' failed during retrieval.", tool_name)

        if calls and failures / len(calls) > 0.5:
            chunks.append("Partial data returned due to tool failures.")

        return {"retrieved_chunks": chunks, "trace_steps": trace_steps}

    return _node
