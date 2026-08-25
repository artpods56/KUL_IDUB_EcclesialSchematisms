"""Add release descriptor digests to Workspace Plugin releases.

The release descriptor references the inspected catalog contract, the
deployment runtime profile, and the artifact invocation protocol as
independent digests. Columns are nullable because rows published before this
migration carry only the embedded catalog JSON; every new publication fills
them. The runtime image digest stays absent until the image-building slice.

Revision ID: 0015_plugin_release_descriptor
Revises: 0014_plugin_releases
Create Date: 2026-08-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0015_plugin_release_descriptor"
down_revision: str | Sequence[str] | None = "0014_plugin_releases"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("plugin_releases") as batch:
        batch.add_column(
            sa.Column("contract_digest", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column("protocol_digest", sa.String(length=64), nullable=True)
        )
        batch.add_column(
            sa.Column("profile_digest", sa.String(length=64), nullable=True)
        )
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_contract_digest",
            "contract_digest IS NULL OR length(contract_digest) = 64",
        )
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_protocol_digest",
            "protocol_digest IS NULL OR length(protocol_digest) = 64",
        )
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_profile_digest",
            "profile_digest IS NULL OR length(profile_digest) = 64",
        )


def downgrade() -> None:
    with op.batch_alter_table("plugin_releases") as batch:
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_profile_digest",
            type_="check",
        )
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_protocol_digest",
            type_="check",
        )
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_contract_digest",
            type_="check",
        )
        batch.drop_column("profile_digest")
        batch.drop_column("protocol_digest")
        batch.drop_column("contract_digest")
