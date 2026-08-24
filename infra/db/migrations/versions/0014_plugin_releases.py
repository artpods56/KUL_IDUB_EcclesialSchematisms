"""Add immutable Workspace Plugin releases.

Revision ID: 0014_plugin_releases
Revises: 0013_thin_execution_schema
Create Date: 2026-08-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0014_plugin_releases"
down_revision: str | Sequence[str] | None = "0013_thin_execution_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "plugin_releases",
        sa.Column("workspace_id", sa.Uuid(), nullable=False),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("catalog", sa.JSON(), nullable=False),
        sa.Column("capabilities", sa.JSON(), nullable=False),
        sa.Column("capability_digest", sa.String(length=64), nullable=False),
        sa.Column("source_object_key", sa.String(length=2048), nullable=False),
        sa.Column("source_digest", sa.String(length=64), nullable=False),
        sa.Column("lock_digest", sa.String(length=64), nullable=False),
        sa.Column("runtime_profile", sa.String(length=100), nullable=False),
        sa.Column("runtime_image_digest", sa.String(length=64), nullable=True),
        sa.Column("published_by_user_id", sa.Uuid(), nullable=True),
        sa.Column("published_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "revision >= 1",
            name="ck_plugin_releases_plugin_release_revision",
        ),
        sa.CheckConstraint(
            "length(capability_digest) = 64",
            name="ck_plugin_releases_plugin_release_capability_digest",
        ),
        sa.CheckConstraint(
            "length(source_digest) = 64",
            name="ck_plugin_releases_plugin_release_source_digest",
        ),
        sa.CheckConstraint(
            "length(lock_digest) = 64",
            name="ck_plugin_releases_plugin_release_lock_digest",
        ),
        sa.CheckConstraint(
            "runtime_image_digest IS NULL OR length(runtime_image_digest) = 64",
            name="ck_plugin_releases_plugin_release_runtime_image_digest",
        ),
        sa.ForeignKeyConstraint(
            ["published_by_user_id"],
            ["users.id"],
            name=op.f("fk_plugin_releases_published_by_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["workspace_id"],
            ["workspaces.id"],
            name=op.f("fk_plugin_releases_workspace_id_workspaces"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "workspace_id",
            "slug",
            "revision",
            name=op.f("pk_plugin_releases"),
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "slug",
            "source_digest",
            name="uq_plugin_releases_workspace_slug_source",
        ),
    )
    op.create_index(
        "ix_plugin_releases_workspace_slug_revision",
        "plugin_releases",
        ["workspace_id", "slug", "revision"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_plugin_releases_workspace_slug_revision",
        table_name="plugin_releases",
    )
    op.drop_table("plugin_releases")
