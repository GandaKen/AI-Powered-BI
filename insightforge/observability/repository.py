"""CRUD and metric-query functions for the local trace store."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import joinedload

from insightforge.db.connection import get_session
from insightforge.observability.models import Trace, TraceStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def get_recent_traces(
    database_url: str,
    limit: int = 50,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent traces, newest first."""
    with get_session(database_url) as session:
        q = (
            session.query(Trace)
            .options(joinedload(Trace.steps))
            .order_by(Trace.created_at.desc())
        )
        if session_id:
            q = q.filter(Trace.session_id == session_id)
        traces = q.limit(limit).all()
        return [t.to_dict() for t in traces]


def get_trace_by_id(database_url: str, trace_id: str) -> dict[str, Any] | None:
    with get_session(database_url) as session:
        trace = (
            session.query(Trace)
            .options(joinedload(Trace.steps))
            .filter(Trace.id == trace_id)
            .first()
        )
        return trace.to_dict() if trace else None


# ---------------------------------------------------------------------------
# Metric aggregations for the Observability dashboard
# ---------------------------------------------------------------------------

def _since(days: int = 30) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


def get_latency_metrics(database_url: str, days: int = 30) -> dict[str, Any]:
    """Avg / P50 / P95 latency and per-step breakdown."""
    since = _since(days)
    with get_session(database_url) as session:
        rows = (
            session.query(Trace.total_latency_ms)
            .filter(Trace.created_at >= since, Trace.total_latency_ms.isnot(None))
            .all()
        )
        latencies = sorted(r[0] for r in rows)
        n = len(latencies)
        if n == 0:
            return {"avg": 0, "p50": 0, "p95": 0, "count": 0, "step_breakdown": []}

        avg = sum(latencies) / n
        p50 = latencies[n // 2]
        p95 = latencies[int(n * 0.95)]

        step_rows = (
            session.query(
                TraceStep.step_name,
                func.avg(TraceStep.latency_ms).label("avg_ms"),
            )
            .join(Trace, TraceStep.trace_id == Trace.id)
            .filter(Trace.created_at >= since)
            .group_by(TraceStep.step_name)
            .order_by(func.avg(TraceStep.latency_ms).desc())
            .all()
        )
        step_breakdown = [
            {"step": r.step_name, "avg_ms": round(float(r.avg_ms), 1)}
            for r in step_rows
        ]

        return {
            "avg": round(avg, 1),
            "p50": p50,
            "p95": p95,
            "count": n,
            "step_breakdown": step_breakdown,
        }


def get_quality_metrics(database_url: str, days: int = 30) -> dict[str, Any]:
    """Quality score trend, error/fallback rates."""
    since = _since(days)
    with get_session(database_url) as session:
        total = (
            session.query(func.count(Trace.id))
            .filter(Trace.created_at >= since)
            .scalar()
        ) or 0

        if total == 0:
            return {
                "avg_quality": None,
                "error_rate": 0.0,
                "fallback_rate": 0.0,
                "total": 0,
            }

        avg_quality = (
            session.query(func.avg(Trace.quality_score))
            .filter(Trace.created_at >= since, Trace.quality_score.isnot(None))
            .scalar()
        )

        error_count = (
            session.query(func.count(Trace.id))
            .filter(Trace.created_at >= since, Trace.status == "error")
            .scalar()
        ) or 0

        fallback_count = (
            session.query(func.count(Trace.id))
            .filter(Trace.created_at >= since, Trace.status == "fallback")
            .scalar()
        ) or 0

        return {
            "avg_quality": round(float(avg_quality), 2) if avg_quality else None,
            "error_rate": round(error_count / total * 100, 1),
            "fallback_rate": round(fallback_count / total * 100, 1),
            "total": total,
        }


def get_usage_metrics(database_url: str, days: int = 30) -> dict[str, Any]:
    """Token consumption and tool usage stats."""
    since = _since(days)
    with get_session(database_url) as session:
        token_totals = (
            session.query(
                func.coalesce(func.sum(Trace.total_tokens_input), 0).label("tokens_in"),
                func.coalesce(func.sum(Trace.total_tokens_output), 0).label("tokens_out"),
            )
            .filter(Trace.created_at >= since)
            .first()
        )

        tool_usage = (
            session.query(
                TraceStep.step_name,
                func.count(TraceStep.id).label("cnt"),
            )
            .join(Trace, TraceStep.trace_id == Trace.id)
            .filter(Trace.created_at >= since)
            .group_by(TraceStep.step_name)
            .all()
        )

        return {
            "total_tokens_input": int(token_totals.tokens_in),
            "total_tokens_output": int(token_totals.tokens_out),
            "tool_usage": {r.step_name: r.cnt for r in tool_usage},
        }


def get_traces_dataframe(database_url: str, days: int = 30) -> pd.DataFrame:
    """Return traces as a DataFrame for Plotly charts."""
    since = _since(days)
    with get_session(database_url) as session:
        rows = (
            session.query(
                Trace.id,
                Trace.query,
                Trace.total_latency_ms,
                Trace.total_tokens_input,
                Trace.total_tokens_output,
                Trace.status,
                Trace.quality_score,
                Trace.model,
                Trace.created_at,
            )
            .filter(Trace.created_at >= since)
            .order_by(Trace.created_at)
            .all()
        )
        if not rows:
            return pd.DataFrame(columns=[
                "id", "query", "total_latency_ms", "total_tokens_input",
                "total_tokens_output", "status", "quality_score", "model", "created_at",
            ])
        return pd.DataFrame(rows, columns=[
            "id", "query", "total_latency_ms", "total_tokens_input",
            "total_tokens_output", "status", "quality_score", "model", "created_at",
        ])


def get_steps_dataframe(database_url: str, days: int = 30) -> pd.DataFrame:
    """Return trace steps as a DataFrame for per-step charts."""
    since = _since(days)
    with get_session(database_url) as session:
        rows = (
            session.query(
                TraceStep.trace_id,
                TraceStep.step_name,
                TraceStep.step_type,
                TraceStep.latency_ms,
                TraceStep.tokens_input,
                TraceStep.tokens_output,
                TraceStep.step_order,
                Trace.created_at,
            )
            .join(Trace, TraceStep.trace_id == Trace.id)
            .filter(Trace.created_at >= since)
            .order_by(Trace.created_at, TraceStep.step_order)
            .all()
        )
        if not rows:
            return pd.DataFrame(columns=[
                "trace_id", "step_name", "step_type", "latency_ms",
                "tokens_input", "tokens_output", "step_order", "created_at",
            ])
        return pd.DataFrame(rows, columns=[
            "trace_id", "step_name", "step_type", "latency_ms",
            "tokens_input", "tokens_output", "step_order", "created_at",
        ])
