"""Answer generation node."""

from __future__ import annotations

import logging

from insightforge.prompts.templates import GENERATOR_PROMPT

logger = logging.getLogger(__name__)

_MAX_GENERATE_RETRIES = 2


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

        prompt = GENERATOR_PROMPT.format(
            context=context,
            question=state.get("query", ""),
        )

        last_err: Exception | None = None
        for attempt in range(_MAX_GENERATE_RETRIES):
            try:
                response = llm.invoke(prompt).content
                return {"generated_response": str(response), "trace_steps": trace_steps}
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "Generator LLM call failed (attempt %d/%d): %s",
                    attempt + 1, _MAX_GENERATE_RETRIES, exc,
                )

        logger.error("Generator exhausted retries; returning error response", exc_info=last_err)
        return {
            "generated_response": (
                "I was unable to generate a response due to a temporary infrastructure "
                "issue. Please try again in a moment."
            ),
            "trace_steps": trace_steps,
        }

    return _node
