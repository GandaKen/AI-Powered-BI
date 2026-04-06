"""Retrieval planning node."""

from __future__ import annotations


def make_retrieval_planner_node():
    """Create retrieval planner node."""

    def _node(state: dict) -> dict:
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append("RetrievalPlanner")

        query = state.get("query", "").lower()
        safety = state.get("safety_check", {})
        if safety.get("label") != "safe":
            strategy = {"blocked": True, "calls": []}
            return {"retrieval_strategy": strategy, "trace_steps": trace_steps}

        calls = [{"tool": "vector_search", "args": {"query": state.get("query", "")}}]
        if any(token in query for token in ["compare", "region", "product", "trend"]):
            calls.append(
                {
                    "tool": "data_analysis",
                    "args": {
                        "request_json": (
                            '{"operation":"groupby_agg","groupby":["Product"],'
                            '"metric":"Sales","agg":"sum"}'
                        ),
                    },
                }
            )
        if any(token in query for token in ["correlation", "percentile", "growth", "yoy"]):
            calls.append(
                {
                    "tool": "statistical",
                    "args": {"request_json": '{"operation":"correlation"}'},
                }
            )

        return {
            "retrieval_strategy": {"blocked": False, "calls": calls},
            "trace_steps": trace_steps,
        }

    return _node
