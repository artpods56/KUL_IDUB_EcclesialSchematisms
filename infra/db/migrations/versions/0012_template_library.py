"""Add the copy-based template library.

Revision ID: 0012_template_library
Revises: 0011_graph_organization
Create Date: 2026-08-11
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0012_template_library"
down_revision: str | Sequence[str] | None = "0011_graph_organization"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "templates",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("source_graph_id", sa.Uuid(), nullable=False),
        sa.Column("source_revision", sa.Integer(), nullable=False),
        sa.Column("source_graph_name", sa.String(length=160), nullable=False),
        sa.Column("snapshot_document", sa.JSON(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("created_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "source_revision >= 1",
            name="ck_templates_template_source_revision",
        ),
        sa.CheckConstraint(
            "state IN ('active', 'archived')",
            name="ck_templates_template_state",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name=op.f("fk_templates_created_by_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_templates_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_templates")),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_templates_workspace_id_id",
        ),
    )
    op.create_index(
        "ix_templates_workspace_name",
        "templates",
        ["workspace_id", "name"],
        unique=False,
    )
    op.create_index(
        "ix_templates_workspace_updated_at",
        "templates",
        ["workspace_id", "updated_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_templates_workspace_updated_at", table_name="templates")
    op.drop_index("ix_templates_workspace_name", table_name="templates")
    op.drop_table("templates")
