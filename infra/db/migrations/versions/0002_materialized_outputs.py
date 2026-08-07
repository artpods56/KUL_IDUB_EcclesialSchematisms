"""Persist artifacts and latest materialized node outputs.

Revision ID: 0002_materialized_outputs
Revises: 0001_saved_graphs
Create Date: 2026-07-15
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0002_materialized_outputs"
down_revision: str | Sequence[str] | None = "0001_saved_graphs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "artifact_objects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("artifact_type", sa.String(length=255), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=False),
        sa.Column("storage_backend", sa.String(length=40), nullable=False),
        sa.Column("bucket", sa.String(length=255), nullable=True),
        sa.Column("object_key", sa.String(length=2048), nullable=True),
        sa.Column("inline_payload", sa.JSON(), nullable=True),
        sa.Column("byte_size", sa.BigInteger(), nullable=True),
        sa.Column("sha256", sa.String(length=64), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_artifact_objects"),
    )
    op.create_index(
        "ix_artifact_objects_sha256",
        "artifact_objects",
        ["sha256"],
        unique=False,
    )
    op.create_index(
        "ix_artifact_objects_type",
        "artifact_objects",
        ["artifact_type", "schema_version"],
        unique=False,
    )

    op.create_table(
        "materialized_node_outputs",
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("graph_revision", sa.Integer(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("workflow_run_id", sa.Uuid(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("materialized_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_id"],
            ["saved_graphs.id"],
            name="fk_materialized_node_outputs_graph_id_saved_graphs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "graph_id",
            "graph_revision",
            "node_id",
            name="pk_materialized_node_outputs",
        ),
    )
    op.create_index(
        "ix_materialized_node_outputs_graph_revision",
        "materialized_node_outputs",
        ["graph_id", "graph_revision", "materialized_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_materialized_node_outputs_graph_revision",
        table_name="materialized_node_outputs",
    )
    op.drop_table("materialized_node_outputs")
    op.drop_index("ix_artifact_objects_type", table_name="artifact_objects")
    op.drop_index("ix_artifact_objects_sha256", table_name="artifact_objects")
    op.drop_table("artifact_objects")
