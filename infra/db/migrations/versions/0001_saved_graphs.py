"""Create saved graph persistence.

Revision ID: 0001_saved_graphs
Revises:
Create Date: 2026-07-13
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0001_saved_graphs"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "saved_graphs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("document", sa.JSON(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_saved_graphs"),
    )
    op.create_index(
        "ix_saved_graphs_updated_at",
        "saved_graphs",
        ["updated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_saved_graphs_updated_at", table_name="saved_graphs")
    op.drop_table("saved_graphs")
