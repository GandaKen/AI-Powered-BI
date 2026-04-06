"""LangChain callback handler that captures per-step trace data for local storage."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from insightforge.db.connection import get_session
from insightforge.observability.models import Trace, TraceStep

logger = logging.getLogger(__name__)

PIPELINE_NODES = frozenset({
    "query_planner",
    "retrieval_planner",
    "information_retriever",
    "context_assembler",
    "generator",
    "response_qa",
})


class _RunInfo:
    """Timing bookkeeping for a single callback run."""

    __slots__ = ("name", "kind", "start", "tokens_in", "tokens_out")

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.start = time.perf_counter()
        self.tokens_in = 0
        self.tokens_out = 0


class TraceCollector(BaseCallbackHandler):
    """Accumulates step-level timing and token data during a single agent invocation.

    Usage::

        collector = TraceCollector(database_url=settings.database_url)
        result = graph.invoke(inputs, config={"callbacks": [collector]})
        collector.finalize(
            query=prompt,
            response=result["final_response"],
            status="success",
            quality_score=result.get("evaluation", {}).get("score"),
            session_id=session_id,
        )
    """

    raise_error = False
    run_inline = True

    def __init__(self, database_url: str, model_name: str = ""):
        super().__init__()
        self._database_url = database_url
        self._model_name = model_name
        self._runs: dict[uuid.UUID, _RunInfo] = {}
        self._completed_steps: list[dict[str, Any]] = []
        self._active_node_run_id: uuid.UUID | None = None
        self._invocation_start = time.perf_counter()
        self._trace_record: Trace | None = None

    # -- chain events (LangGraph nodes fire as chains) -----------------------

    def on_chain_start(
        self, serialized: dict[str, Any] | None, inputs: dict[str, Any] | None, *, run_id, **kwargs
    ) -> None:
        name = (serialized or {}).get("name", "").lower()
        if name in PIPELINE_NODES:
            info = _RunInfo(name, "chain")
            self._runs[run_id] = info
            self._active_node_run_id = run_id

    def on_chain_end(self, outputs: dict[str, Any], *, run_id, **kwargs) -> None:
        info = self._runs.pop(run_id, None)
        if info is None:
            return
        elapsed = int((time.perf_counter() - info.start) * 1000)
        output_summary = ""
        if isinstance(outputs, dict):
            for key in ("generated_response", "assembled_context", "safety_check"):
                if key in outputs:
                    val = str(outputs[key])
                    output_summary = val[:500]
                    break

        self._completed_steps.append({
            "step_name": info.name,
            "step_type": info.kind,
            "latency_ms": elapsed,
            "tokens_input": info.tokens_in,
            "tokens_output": info.tokens_out,
            "output_summary": output_summary,
        })
        if run_id == self._active_node_run_id:
            self._active_node_run_id = None

    def on_chain_error(self, error: BaseException, *, run_id, **kwargs) -> None:
        info = self._runs.pop(run_id, None)
        if info is None:
            return
        elapsed = int((time.perf_counter() - info.start) * 1000)
        self._completed_steps.append({
            "step_name": info.name,
            "step_type": info.kind,
            "latency_ms": elapsed,
            "tokens_input": info.tokens_in,
            "tokens_output": info.tokens_out,
            "output_summary": f"ERROR: {error!r}"[:500],
            "metadata": {"error": True},
        })

    # -- LLM events (token tracking) -----------------------------------------

    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: list[str], *, run_id, **kwargs) -> None:
        self._runs[run_id] = _RunInfo((serialized or {}).get("name", "llm"), "llm")

    def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        llm_info = self._runs.pop(run_id, None)
        usage = (response.llm_output or {}).get("token_usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        if self._active_node_run_id and self._active_node_run_id in self._runs:
            parent = self._runs[self._active_node_run_id]
            parent.tokens_in += tokens_in
            parent.tokens_out += tokens_out
        elif llm_info:
            llm_info.tokens_in = tokens_in
            llm_info.tokens_out = tokens_out

    # -- tool events ----------------------------------------------------------

    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, *, run_id, **kwargs) -> None:
        self._runs[run_id] = _RunInfo((serialized or {}).get("name", "tool"), "tool")

    def on_tool_end(self, output: str, *, run_id, **kwargs) -> None:
        self._runs.pop(run_id, None)

    # -- public API -----------------------------------------------------------

    @property
    def steps(self) -> list[dict[str, Any]]:
        return list(self._completed_steps)

    @property
    def total_latency_ms(self) -> int:
        return int((time.perf_counter() - self._invocation_start) * 1000)

    def finalize(
        self,
        *,
        query: str,
        response: str,
        status: str = "success",
        quality_score: float | None = None,
        session_id: str = "",
        langfuse_trace_id: str = "",
        langfuse_url: str = "",
    ) -> dict[str, Any] | None:
        """Persist the accumulated trace to Postgres and return its dict representation."""
        total_latency = self.total_latency_ms
        total_in = sum(s.get("tokens_input", 0) for s in self._completed_steps)
        total_out = sum(s.get("tokens_output", 0) for s in self._completed_steps)

        trace = Trace(
            session_id=session_id,
            query=query,
            response=response[:5000] if response else "",
            total_latency_ms=total_latency,
            total_tokens_input=total_in,
            total_tokens_output=total_out,
            model=self._model_name,
            status=status,
            quality_score=quality_score,
            langfuse_trace_id=langfuse_trace_id,
            langfuse_url=langfuse_url,
        )

        for order, step_data in enumerate(self._completed_steps):
            trace.steps.append(TraceStep(
                step_name=step_data["step_name"],
                step_type=step_data.get("step_type", "chain"),
                output_summary=step_data.get("output_summary", ""),
                latency_ms=step_data.get("latency_ms", 0),
                tokens_input=step_data.get("tokens_input", 0),
                tokens_output=step_data.get("tokens_output", 0),
                step_metadata=step_data.get("metadata", {}),
                step_order=order,
            ))

        try:
            with get_session(self._database_url) as session:
                session.add(trace)
                session.flush()
                self._trace_record = trace
                result = trace.to_dict()
            return result
        except Exception:
            logger.exception("Failed to persist trace to Postgres")
            return None
