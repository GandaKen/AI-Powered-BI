"""Answer generation node."""

from __future__ import annotations

from insightforge.prompts.templates import GENERATOR_PROMPT


def make_generator_node(llm):
    """Create response generator node."""

    def _node(state: dict) -> dict:
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append("Generator")

        safety = state.get("safety_check", {})
        if safety.get("label") != "safe":
            return {
                "generated_response": (
                    "I cannot process that request because it appears unsafe. "
                    "Please ask a sales-data-focused question."
                ),
                "trace_steps": trace_steps,
            }

        context = state.get("assembled_context", "").strip()
        if not context:
            return {
                "generated_response": (
                    "I could not retrieve enough data context to answer confidently."
                ),
                "trace_steps": trace_steps,
            }

        response = llm.invoke(
            GENERATOR_PROMPT.format(
                context=context,
                question=state.get("query", ""),
            )
        ).content

        return {"generated_response": str(response), "trace_steps": trace_steps}

    return _node

