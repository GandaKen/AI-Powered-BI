"""Context assembly node."""

from __future__ import annotations


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def make_context_assembler_node(settings):
    """Create context assembler node."""

    def _node(state: dict) -> dict:
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append("ContextAssembler")

        plan = state.get("query_plan", {})
        complexity = plan.get("complexity", "simple")
        budget = (
            settings.token_budget_complex
            if complexity == "compound"
            else settings.token_budget_simple
        )

        unique_chunks = []
        seen = set()
        for chunk in state.get("retrieved_chunks", []):
            if chunk not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk)

        assembled_parts = []
        used_tokens = 0
        for chunk in unique_chunks:
            chunk_tokens = _estimate_tokens(chunk)
            if used_tokens + chunk_tokens > budget:
                break
            assembled_parts.append(chunk)
            used_tokens += chunk_tokens

        return {
            "assembled_context": "\n\n".join(assembled_parts),
            "trace_steps": trace_steps,
        }

    return _node
