"""Add immutable OCI artifact references to Workspace Plugin releases.

Existing releases remain source-only with null runtime fields. New publication
uses a descriptor digest, allowing the same source bytes to produce a distinct
append-only release when deployment-owned runtime inputs change.

Revision ID: 0017_plugin_runtime_artifact
Revises: 0016_execution_node_diagnostics
Create Date: 2026-08-24
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0017_plugin_runtime_artifact"
down_revision: str | Sequence[str] | None = "0016_execution_node_diagnostics"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("plugin_releases") as batch:
        batch.drop_constraint(
            "uq_plugin_releases_workspace_slug_source",
            type_="unique",
        )
        batch.add_column(sa.Column("runtime_artifact", sa.JSON(), nullable=True))
        batch.add_column(
            sa.Column("descriptor_digest", sa.String(length=64), nullable=True)
        )
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_descriptor_digest",
            "descriptor_digest IS NULL OR length(descriptor_digest) = 64",
        )
        batch.create_unique_constraint(
            "uq_plugin_releases_workspace_slug_descriptor",
            ("workspace_id", "slug", "descriptor_digest"),
        )


def downgrade() -> None:
    with op.batch_alter_table("plugin_releases") as batch:
        batch.drop_constraint(
            "uq_plugin_releases_workspace_slug_descriptor",
            type_="unique",
        )
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_descriptor_digest",
            type_="check",
        )
        batch.drop_column("descriptor_digest")
        batch.drop_column("runtime_artifact")
        batch.create_unique_constraint(
            "uq_plugin_releases_workspace_slug_source",
            ("workspace_id", "slug", "source_digest"),
        )
