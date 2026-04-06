"""Input safety classifier for prompt-injection resilience."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_BLOCKED_TOKENS = [
    "ignore previous instructions",
    "system prompt",
    "jailbreak",
    "os.system",
    "subprocess",
    "rm -rf",
    "drop table",
    "exec(",
    "eval(",
    "__import__",
]


@dataclass
class GuardrailResult:
    """Classification result for a user query."""

    label: str
    reason: str

    @property
    def is_safe(self) -> bool:
        return self.label == "safe"


class QueryGuardrail:
    """Query guardrail with fast heuristic and optional LLM fallback.

    The heuristic check is always applied first. An LLM call is only made
    when *use_llm* is True and the heuristic did not already block the query.
    Keeping *use_llm=False* (default) eliminates a full LLM round-trip per
    request — a major latency saving in the agentic pipeline.
    """

    def __init__(self, llm, *, use_llm: bool = False) -> None:
        self._llm = llm
        self._use_llm = use_llm

    def classify(self, query: str) -> GuardrailResult:
        lower_query = query.lower()
        if any(token in lower_query for token in _BLOCKED_TOKENS):
            return GuardrailResult(
                label="injection_attempt",
                reason="Detected prompt-injection or unsafe execution pattern.",
            )

        if not self._use_llm:
            return GuardrailResult(label="safe", reason="Heuristic pass.")

        prompt = (
            "Classify this message for BI assistant safety. "
            "Return JSON with keys: label (safe|injection_attempt|off_topic), reason.\n\n"
            f"Message: {query}"
        )
        try:
            raw = self._llm.invoke(prompt).content
            payload = json.loads(raw) if isinstance(raw, str) else {}
            label = payload.get("label", "safe")
            reason = payload.get("reason", "No risk found.")
            return GuardrailResult(label=label, reason=reason)
        except Exception:
            logger.debug("LLM guardrail failed; falling back to heuristic pass", exc_info=True)
            return GuardrailResult(label="safe", reason="Guardrail fallback path used.")
