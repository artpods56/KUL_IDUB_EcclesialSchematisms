"""Persist the submitted request for durable local execution dispatch.

Revision ID: 0018_local_execution_queue
Revises: 0017_plugin_runtime_artifact
Create Date: 2026-08-24
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0018_local_execution_queue"
down_revision: str | Sequence[str] | None = "0017_plugin_runtime_artifact"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("graph_executions") as batch:
        batch.add_column(sa.Column("submitted_request", sa.JSON(), nullable=True))
        batch.add_column(
            sa.Column("idempotency_key", sa.String(length=255), nullable=True)
        )
        batch.add_column(sa.Column("submitted_by_actor_id", sa.Uuid(), nullable=True))
        batch.create_unique_constraint(
            "uq_graph_executions_workspace_idempotency_key",
            ("workspace_id", "idempotency_key"),
        )
        batch.create_index(
            "ix_graph_executions_queue_order",
            ("status", "created_at", "execution_id"),
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("graph_executions") as batch:
        batch.drop_index("ix_graph_executions_queue_order")
        batch.drop_constraint(
            "uq_graph_executions_workspace_idempotency_key",
            type_="unique",
        )
        batch.drop_column("idempotency_key")
        batch.drop_column("submitted_by_actor_id")
        batch.drop_column("submitted_request")
