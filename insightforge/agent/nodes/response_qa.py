"""Merged response evaluation and refinement node."""

from __future__ import annotations

import json

from insightforge.prompts.templates import RESPONSE_QA_PROMPT


def _heuristic_score(response: str) -> tuple[int, list[str]]:
    issues: list[str] = []
    score = 8
    if not response or len(response.strip()) < 20:
        score -= 3
        issues.append("Response is too short.")
    if "could not" in response.lower() or "sorry" in response.lower():
        score -= 2
        issues.append("Low confidence response.")
    return max(1, score), issues


def make_response_qa_node(llm, settings):
    """Create response QA node with at most one retry."""

    def _node(state: dict) -> dict:
        trace_steps = list(state.get("trace_steps", []))
        trace_steps.append("ResponseQA")

        response = state.get("generated_response", "")
        score, issues = _heuristic_score(response)

        refined = response
        try:
            raw = llm.invoke(
                RESPONSE_QA_PROMPT.format(
                    question=state.get("query", ""),
                    response=response,
                )
            ).content
            parsed = json.loads(raw) if isinstance(raw, str) else {}
            refined = parsed.get("refined_response", response)
            score = int(parsed.get("score", score))
            issues = parsed.get("issues", issues)
        except Exception:
            pass

        retry_count = state.get("retry_count", 0)
        needs_retry = score < 7 and retry_count < settings.max_eval_retries
        next_retry_count = retry_count + 1 if needs_retry else retry_count

        return {
            "evaluation": {
                "score": score,
                "issues": issues,
                "needs_retry": needs_retry,
            },
            "retry_count": next_retry_count,
            "final_response": refined,
            "trace_steps": trace_steps,
        }

    return _node

