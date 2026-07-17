"""Persist encrypted node secrets.

Revision ID: 0003_node_secrets
Revises: 0002_materialized_outputs
Create Date: 2026-07-16
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0003_node_secrets"
down_revision: str | Sequence[str] | None = "0002_materialized_outputs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "node_secrets",
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("node_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("operator_id", sa.String(length=255), nullable=False),
        sa.Column("operator_version", sa.Integer(), nullable=False),
        sa.Column("key_id", sa.String(length=64), nullable=False),
        sa.Column("dependency_sha256", sa.String(length=64), nullable=False),
        sa.Column("nonce", sa.LargeBinary(length=12), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_id"],
            ["saved_graphs.id"],
            name="fk_node_secrets_graph_id_saved_graphs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "graph_id",
            "node_id",
            "name",
            name="pk_node_secrets",
        ),
    )
    op.create_index(
        "ix_node_secrets_graph_id",
        "node_secrets",
        ["graph_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_node_secrets_graph_id", table_name="node_secrets")
    op.drop_table("node_secrets")
