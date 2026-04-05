"""Query planning node (merged analyzer + orchestrator)."""

from __future__ import annotations

import json
import logging

from insightforge.prompts.templates import QUERY_PLANNER_PROMPT

logger = logging.getLogger(__name__)


def make_query_planner_node(llm, guardrail):
    """Create query planner node."""

    def _node(state: dict) -> dict:
        query = state.get("query", "")
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append("QueryPlanner")

        safety = guardrail.classify(query)
        query_plan = {
            "intent": "factual",
            "complexity": "simple",
            "entities": [],
            "tasks": [query],
        }
        if safety.is_safe:
            try:
                raw = llm.invoke(QUERY_PLANNER_PROMPT.format(query=query)).content
                query_plan = json.loads(raw) if isinstance(raw, str) else query_plan
            except Exception:
                logger.warning("Failed to parse query plan for: %s", query[:80], exc_info=True)

        return {
            "safety_check": {"label": safety.label, "reason": safety.reason},
            "query_plan": query_plan,
            "retry_count": state.get("retry_count", 0),
            "trace_steps": trace_steps,
        }

    return _node

