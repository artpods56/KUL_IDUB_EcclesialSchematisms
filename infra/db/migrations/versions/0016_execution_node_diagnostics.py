"""Add retained execution diagnostics to graph execution node results.

The nullable diagnostics column stores metadata-only provenance for one node
result, such as the exact Workspace Plugin release identity (slug, revision,
source digest) that a compiled node resolved to. It never stores credentials,
provider payloads, or command bodies.

Revision ID: 0016_execution_node_diagnostics
Revises: 0015_plugin_release_descriptor
Create Date: 2026-08-23
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0016_execution_node_diagnostics"
down_revision: str | Sequence[str] | None = "0015_plugin_release_descriptor"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("graph_execution_nodes") as batch:
        batch.add_column(sa.Column("diagnostics", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("graph_execution_nodes") as batch:
        batch.drop_column("diagnostics")
