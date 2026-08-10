"""Add workspace module library tables.

Revision ID: 0010_module_library
Revises: 0009_collaborative_graph_heads
Create Date: 2026-08-08
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0010_module_library"
down_revision: str | Sequence[str] | None = "0009_collaborative_graph_heads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "modules",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("source_graph_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("description", sa.String(length=1000), nullable=True),
        sa.Column("publication_state", sa.String(length=32), nullable=False),
        sa.Column("current_library_release", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "publication_state IN ('published', 'deprecated', 'withdrawn')",
            name="ck_modules_module_publication_state",
        ),
        sa.CheckConstraint(
            "current_library_release IS NULL OR current_library_release >= 1",
            name="ck_modules_module_current_library_release",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "source_graph_id"],
            ["saved_graphs.workspace_id", "saved_graphs.id"],
            name=op.f("fk_modules_source_graph_id_saved_graphs"),
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_modules_workspace_id_workspaces"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_modules")),
        sa.UniqueConstraint(
            "workspace_id",
            "id",
            name="uq_modules_workspace_id_id",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "source_graph_id",
            name="uq_modules_workspace_source_graph",
        ),
    )
    op.create_index(
        "ix_modules_workspace_updated_at",
        "modules",
        ["workspace_id", "updated_at"],
        unique=False,
    )

    op.create_table(
        "module_releases",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("module_id", sa.Uuid(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source_graph_id", sa.Uuid(), nullable=False),
        sa.Column("published_at", sa.DateTime(), nullable=False),
        sa.Column("published_by_user_id", sa.Uuid(), nullable=True),
        sa.CheckConstraint(
            "revision >= 1",
            name="ck_module_releases_module_release_revision",
        ),
        sa.ForeignKeyConstraint(
            ["published_by_user_id"],
            ["users.id"],
            name=op.f("fk_module_releases_published_by_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "module_id"],
            ["modules.workspace_id", "modules.id"],
            name=op.f("fk_module_releases_module_id_modules"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id", "source_graph_id", "revision"],
            [
                "saved_graph_revisions.workspace_id",
                "saved_graph_revisions.graph_id",
                "saved_graph_revisions.revision",
            ],
            name="fk_module_releases_saved_graph_revision",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "module_id",
            "revision",
            name=op.f("pk_module_releases"),
        ),
    )
    op.create_index(
        "ix_module_releases_workspace_module_revision",
        "module_releases",
        ["workspace_id", "module_id", "revision"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_module_releases_workspace_module_revision",
        table_name="module_releases",
    )
    op.drop_table("module_releases")
    op.drop_index("ix_modules_workspace_updated_at", table_name="modules")
    op.drop_table("modules")
