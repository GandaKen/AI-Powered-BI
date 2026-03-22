"""Create trace tables.

Revision ID: 001
Revises:
Create Date: 2026-03-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", sa.String(255), index=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("response", sa.Text()),
        sa.Column("total_latency_ms", sa.Integer()),
        sa.Column("total_tokens_input", sa.Integer(), server_default="0"),
        sa.Column("total_tokens_output", sa.Integer(), server_default="0"),
        sa.Column("model", sa.String(255)),
        sa.Column("status", sa.String(50), server_default="success"),
        sa.Column("quality_score", sa.Float()),
        sa.Column("langfuse_trace_id", sa.String(255)),
        sa.Column("langfuse_url", sa.Text()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            index=True,
        ),
    )

    op.create_table(
        "trace_steps",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "trace_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("traces.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("step_name", sa.String(255), nullable=False),
        sa.Column("step_type", sa.String(50)),
        sa.Column("input_summary", sa.Text()),
        sa.Column("output_summary", sa.Text()),
        sa.Column("latency_ms", sa.Integer()),
        sa.Column("tokens_input", sa.Integer(), server_default="0"),
        sa.Column("tokens_output", sa.Integer(), server_default="0"),
        sa.Column("metadata", postgresql.JSONB(), server_default="{}"),
        sa.Column("step_order", sa.Integer(), server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )


def downgrade() -> None:
    op.drop_table("trace_steps")
    op.drop_table("traces")
