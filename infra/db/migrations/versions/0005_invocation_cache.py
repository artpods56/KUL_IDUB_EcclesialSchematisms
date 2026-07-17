"""Persist content-addressed node invocation cache entries.

Revision ID: 0005_invocation_cache
Revises: 0004_saved_graph_revisions
Create Date: 2026-07-16
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0005_invocation_cache"
down_revision: str | Sequence[str] | None = "0004_saved_graph_revisions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "invocation_cache_entries",
        sa.Column("key_sha256", sa.String(length=64), nullable=False),
        sa.Column("generation", sa.Uuid(), nullable=False),
        sa.Column("outputs", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint(
            "key_sha256",
            name="pk_invocation_cache_entries",
        ),
    )


def downgrade() -> None:
    op.drop_table("invocation_cache_entries")
