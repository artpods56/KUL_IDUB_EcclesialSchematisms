"""Persist append-only graph execution history.

Revision ID: 0006_execution_history
Revises: 0005_invocation_cache
Create Date: 2026-07-18
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0006_execution_history"
down_revision: str | Sequence[str] | None = "0005_invocation_cache"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "graph_executions",
        sa.Column("execution_id", sa.Uuid(), nullable=False),
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("graph_revision", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("scope", sa.String(length=32), nullable=False),
        sa.Column("workflow_run_id", sa.Uuid(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["graph_id", "graph_revision"],
            ["saved_graph_revisions.graph_id", "saved_graph_revisions.revision"],
            name="fk_graph_executions_graph_id_saved_graph_revisions",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "execution_id",
            name="pk_graph_executions",
        ),
    )
    op.create_index(
        "ix_graph_executions_graph_created",
        "graph_executions",
        ["graph_id", "created_at", "execution_id"],
        unique=False,
    )
    op.create_index(
        "ix_graph_executions_graph_revision_created",
        "graph_executions",
        ["graph_id", "graph_revision", "created_at", "execution_id"],
        unique=False,
    )
    op.create_index(
        "ix_graph_executions_status",
        "graph_executions",
        ["status"],
        unique=False,
    )

    op.create_table(
        "graph_execution_requested_nodes",
        sa.Column("execution_id", sa.Uuid(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            ["graph_executions.execution_id"],
            name=("fk_exec_req_nodes_execution"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "execution_id",
            "node_id",
            name="pk_graph_execution_requested_nodes",
        ),
        sa.UniqueConstraint(
            "execution_id",
            "position",
            name="uq_graph_execution_requested_nodes_execution_position",
        ),
    )
    op.create_index(
        "ix_graph_execution_requested_nodes_node_execution",
        "graph_execution_requested_nodes",
        ["node_id", "execution_id"],
        unique=False,
    )

    op.create_table(
        "graph_execution_node_results",
        sa.Column("execution_id", sa.Uuid(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("artifact_count", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["execution_id"],
            ["graph_executions.execution_id"],
            name=("fk_exec_result_nodes_execution"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "execution_id",
            "node_id",
            name="pk_graph_execution_node_results",
        ),
        sa.UniqueConstraint(
            "execution_id",
            "position",
            name="uq_graph_execution_node_results_execution_position",
        ),
    )
    op.create_index(
        "ix_graph_execution_node_results_node_execution",
        "graph_execution_node_results",
        ["node_id", "execution_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_graph_execution_node_results_node_execution",
        table_name="graph_execution_node_results",
    )
    op.drop_table("graph_execution_node_results")
    op.drop_index(
        "ix_graph_execution_requested_nodes_node_execution",
        table_name="graph_execution_requested_nodes",
    )
    op.drop_table("graph_execution_requested_nodes")
    op.drop_index(
        "ix_graph_executions_status",
        table_name="graph_executions",
    )
    op.drop_index(
        "ix_graph_executions_graph_revision_created",
        table_name="graph_executions",
    )
    op.drop_index(
        "ix_graph_executions_graph_created",
        table_name="graph_executions",
    )
    op.drop_table("graph_executions")
