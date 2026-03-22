"""SQLAlchemy ORM models for local trace storage."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Trace(Base):
    __tablename__ = "traces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(255), index=True)
    query = Column(Text, nullable=False)
    response = Column(Text)
    total_latency_ms = Column(Integer)
    total_tokens_input = Column(Integer, default=0)
    total_tokens_output = Column(Integer, default=0)
    model = Column(String(255))
    status = Column(String(50), default="success")
    quality_score = Column(Float)
    langfuse_trace_id = Column(String(255))
    langfuse_url = Column(Text)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    steps = relationship(
        "TraceStep",
        back_populates="trace",
        cascade="all, delete-orphan",
        order_by="TraceStep.step_order",
    )

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "session_id": self.session_id,
            "query": self.query,
            "response": self.response,
            "total_latency_ms": self.total_latency_ms,
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "model": self.model,
            "status": self.status,
            "quality_score": self.quality_score,
            "langfuse_trace_id": self.langfuse_trace_id,
            "langfuse_url": self.langfuse_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "steps": [s.to_dict() for s in (self.steps or [])],
        }


class TraceStep(Base):
    __tablename__ = "trace_steps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("traces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_name = Column(String(255), nullable=False)
    step_type = Column(String(50))
    input_summary = Column(Text)
    output_summary = Column(Text)
    latency_ms = Column(Integer)
    tokens_input = Column(Integer, default=0)
    tokens_output = Column(Integer, default=0)
    step_metadata = Column("metadata", JSONB, default=dict)
    step_order = Column(Integer, default=0)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    trace = relationship("Trace", back_populates="steps")

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "step_name": self.step_name,
            "step_type": self.step_type,
            "input_summary": self.input_summary,
            "output_summary": self.output_summary,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "metadata": self.step_metadata,
            "step_order": self.step_order,
        }
