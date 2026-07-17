"""Persist immutable saved graph revision snapshots.

Revision ID: 0004_saved_graph_revisions
Revises: 0003_node_secrets
Create Date: 2026-07-16
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0004_saved_graph_revisions"
down_revision: str | Sequence[str] | None = "0003_node_secrets"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "saved_graph_revisions",
        sa.Column("graph_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["graph_id"],
            ["saved_graphs.id"],
            name="fk_saved_graph_revisions_graph_id_saved_graphs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "graph_id",
            "revision",
            name="pk_saved_graph_revisions",
        ),
    )

    saved_graphs = sa.table(
        "saved_graphs",
        sa.column("id", sa.Uuid()),
        sa.column("name", sa.String(length=160)),
        sa.column("document", sa.JSON()),
        sa.column("revision", sa.Integer()),
        sa.column("updated_at", sa.DateTime()),
    )
    saved_graph_revisions = sa.table(
        "saved_graph_revisions",
        sa.column("graph_id", sa.Uuid()),
        sa.column("revision", sa.Integer()),
        sa.column("name", sa.String(length=160)),
        sa.column("document", sa.JSON()),
        sa.column("created_at", sa.DateTime()),
    )
    op.execute(
        saved_graph_revisions.insert().from_select(
            ["graph_id", "revision", "name", "document", "created_at"],
            sa.select(
                saved_graphs.c.id,
                saved_graphs.c.revision,
                saved_graphs.c.name,
                saved_graphs.c.document,
                saved_graphs.c.updated_at,
            ),
        )
    )


def downgrade() -> None:
    op.drop_table("saved_graph_revisions")
