"""Add scoped System and Workspace Plugin release identities.

Revision ID: 0019_scoped_plugin_releases
Revises: 0018_local_execution_queue
Create Date: 2026-08-24
"""

from collections.abc import Sequence
from typing import cast
from uuid import uuid4

from alembic import op
import sqlalchemy as sa


revision: str = "0019_scoped_plugin_releases"
down_revision: str | Sequence[str] | None = "0018_local_execution_queue"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_GRAPH_DOCUMENT_COLUMNS = (
    ("saved_graphs", "document"),
    ("saved_graph_revisions", "document"),
    ("collaborative_graph_heads", "document"),
    ("templates", "snapshot_document"),
)
_RETAINED_REFERENCE_COLUMNS = (
    ("graph_executions", "submitted_request"),
    ("artifact_objects", "metadata"),
    ("graph_execution_nodes", "diagnostics"),
)
_JSON_COLUMNS = _GRAPH_DOCUMENT_COLUMNS + _RETAINED_REFERENCE_COLUMNS
_RELEASE_REFERENCE_KEYS = frozenset({"plugin_release", "plugin_release_pin"})

_SYSTEM_REVISION_INDEX = "uq_plugin_releases_system_slug_revision"
_WORKSPACE_REVISION_INDEX = "uq_plugin_releases_workspace_slug_revision"
_SYSTEM_DESCRIPTOR_INDEX = "uq_plugin_releases_system_slug_descriptor"
_WORKSPACE_DESCRIPTOR_INDEX = "uq_plugin_releases_workspace_slug_descriptor"


def _transform_release_references(
    value: object,
    *,
    add_workspace_scope: bool,
) -> tuple[object, bool, bool]:
    """Return transformed JSON, whether it changed, and whether it pins System."""

    if isinstance(value, list):
        changed = False
        contains_system = False
        transformed_items: list[object] = []
        for item in cast(list[object], value):
            transformed, item_changed, item_contains_system = (
                _transform_release_references(
                    item,
                    add_workspace_scope=add_workspace_scope,
                )
            )
            transformed_items.append(transformed)
            changed = changed or item_changed
            contains_system = contains_system or item_contains_system
        return transformed_items, changed, contains_system

    if not isinstance(value, dict):
        return value, False, False

    changed = False
    contains_system = False
    transformed_mapping: dict[object, object] = {}
    mapping = cast(dict[object, object], value)
    for key, item in mapping.items():
        transformed, item_changed, item_contains_system = _transform_release_references(
            item,
            add_workspace_scope=add_workspace_scope,
        )
        changed = changed or item_changed
        contains_system = contains_system or item_contains_system

        if (
            isinstance(key, str)
            and key in _RELEASE_REFERENCE_KEYS
            and isinstance(transformed, dict)
            and "slug" in transformed
            and "revision" in transformed
        ):
            release_reference = dict(cast(dict[object, object], transformed))
            scope = release_reference.get("scope")
            contains_system = contains_system or scope == "system"
            if add_workspace_scope and scope is None:
                release_reference["scope"] = "workspace"
                changed = True
            elif not add_workspace_scope and scope == "workspace":
                release_reference.pop("scope")
                changed = True
            transformed = release_reference
        transformed_mapping[key] = transformed
    return transformed_mapping, changed, contains_system


def _json_tables(
    connection: sa.Connection,
    columns: tuple[tuple[str, str], ...],
) -> list[tuple[sa.Table, str]]:
    inspector = sa.inspect(connection)
    table_names = set(inspector.get_table_names())
    tables: list[tuple[sa.Table, str]] = []
    for table_name, column_name in columns:
        if table_name not in table_names:
            continue
        column_names = {column["name"] for column in inspector.get_columns(table_name)}
        if column_name not in column_names:
            continue
        tables.append(
            (
                sa.Table(
                    table_name,
                    sa.MetaData(),
                    autoload_with=connection,
                ),
                column_name,
            )
        )
    return tables


def _system_reference_locations(
    connection: sa.Connection,
    tables: list[tuple[sa.Table, str]],
) -> list[str]:
    locations: list[str] = []
    for table, column_name in tables:
        column = table.c[column_name]
        for value in connection.execute(sa.select(column)).scalars():
            if value is None:
                continue
            _, _, contains_system = _transform_release_references(
                value,
                add_workspace_scope=False,
            )
            if contains_system:
                locations.append(f"{table.name}.{column_name}")
                break
    return locations


def _rewrite_release_references(
    connection: sa.Connection,
    tables: list[tuple[sa.Table, str]],
    *,
    add_workspace_scope: bool,
    root_schema_version: int | None = None,
) -> None:
    for table, column_name in tables:
        primary_key_columns = tuple(table.primary_key.columns)
        column = table.c[column_name]
        rows = connection.execute(sa.select(*primary_key_columns, column)).mappings()
        for row in rows:
            value = row[column_name]
            if value is None:
                continue
            transformed, changed, _ = _transform_release_references(
                value,
                add_workspace_scope=add_workspace_scope,
            )
            if root_schema_version is not None:
                if not isinstance(transformed, dict):
                    raise RuntimeError(
                        f"Cannot migrate {table.name}.{column_name}: "
                        "graph document root is not a JSON object"
                    )
                document = dict(cast(dict[object, object], transformed))
                if document.get("schema_version") != root_schema_version:
                    document["schema_version"] = root_schema_version
                    changed = True
                transformed = document
            if not changed:
                continue
            identity = sa.and_(
                *(
                    primary_key == row[primary_key.name]
                    for primary_key in primary_key_columns
                )
            )
            connection.execute(
                sa.update(table).where(identity).values({column_name: transformed})
            )


def _backfill_release_identity(connection: sa.Connection) -> None:
    releases = sa.table(
        "plugin_releases",
        sa.column("id", sa.Uuid()),
        sa.column("scope", sa.String(length=16)),
        sa.column("workspace_id", sa.Uuid()),
        sa.column("slug", sa.String(length=100)),
        sa.column("revision", sa.Integer()),
        sa.column("execution_policy", sa.String(length=24)),
        sa.column("distribution", sa.String(length=16)),
    )
    rows = connection.execute(
        sa.select(
            releases.c.workspace_id,
            releases.c.slug,
            releases.c.revision,
        )
    ).all()
    for workspace_id, slug, release_revision in rows:
        connection.execute(
            sa.update(releases)
            .where(
                releases.c.workspace_id == workspace_id,
                releases.c.slug == slug,
                releases.c.revision == release_revision,
            )
            .values(
                id=uuid4(),
                scope="workspace",
                execution_policy="isolated-only",
                distribution=None,
            )
        )


def _create_scoped_indexes() -> None:
    op.create_index(
        _SYSTEM_REVISION_INDEX,
        "plugin_releases",
        ["slug", "revision"],
        unique=True,
        sqlite_where=sa.text("scope = 'system'"),
        postgresql_where=sa.text("scope = 'system'"),
    )
    op.create_index(
        _WORKSPACE_REVISION_INDEX,
        "plugin_releases",
        ["workspace_id", "slug", "revision"],
        unique=True,
        sqlite_where=sa.text("scope = 'workspace'"),
        postgresql_where=sa.text("scope = 'workspace'"),
    )
    op.create_index(
        _SYSTEM_DESCRIPTOR_INDEX,
        "plugin_releases",
        ["slug", "descriptor_digest"],
        unique=True,
        sqlite_where=sa.text("scope = 'system' AND descriptor_digest IS NOT NULL"),
        postgresql_where=sa.text("scope = 'system' AND descriptor_digest IS NOT NULL"),
    )
    op.create_index(
        _WORKSPACE_DESCRIPTOR_INDEX,
        "plugin_releases",
        ["workspace_id", "slug", "descriptor_digest"],
        unique=True,
        sqlite_where=sa.text("scope = 'workspace' AND descriptor_digest IS NOT NULL"),
        postgresql_where=sa.text(
            "scope = 'workspace' AND descriptor_digest IS NOT NULL"
        ),
    )


def upgrade() -> None:
    connection = op.get_bind()
    with op.batch_alter_table("plugin_releases") as batch:
        batch.add_column(sa.Column("id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("scope", sa.String(length=16), nullable=True))
        batch.add_column(
            sa.Column("execution_policy", sa.String(length=24), nullable=True)
        )
        batch.add_column(sa.Column("distribution", sa.String(length=16), nullable=True))

    _backfill_release_identity(connection)

    op.drop_index(
        "ix_plugin_releases_workspace_slug_revision",
        table_name="plugin_releases",
    )
    with op.batch_alter_table("plugin_releases") as batch:
        batch.drop_constraint(
            "uq_plugin_releases_workspace_slug_descriptor",
            type_="unique",
        )
        batch.drop_constraint("pk_plugin_releases", type_="primary")
        batch.alter_column("id", existing_type=sa.Uuid(), nullable=False)
        batch.alter_column(
            "scope",
            existing_type=sa.String(length=16),
            nullable=False,
        )
        batch.alter_column(
            "execution_policy",
            existing_type=sa.String(length=24),
            nullable=False,
        )
        batch.alter_column(
            "workspace_id",
            existing_type=sa.Uuid(),
            nullable=True,
        )
        batch.create_primary_key("pk_plugin_releases", ["id"])
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_scope",
            "scope IN ('system', 'workspace')",
        )
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_scope_workspace",
            "(scope = 'system' AND workspace_id IS NULL) OR "
            "(scope = 'workspace' AND workspace_id IS NOT NULL)",
        )
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_execution_policy",
            "execution_policy IN ('host-eligible', 'isolated-only')",
        )
        batch.create_check_constraint(
            "ck_plugin_releases_plugin_release_scope_policy",
            "(scope = 'system' AND distribution IN "
            "('bundled', 'optional', 'published')) OR "
            "(scope = 'workspace' AND distribution IS NULL "
            "AND execution_policy = 'isolated-only')",
        )

    _create_scoped_indexes()
    _rewrite_release_references(
        connection,
        _json_tables(connection, _GRAPH_DOCUMENT_COLUMNS),
        add_workspace_scope=True,
        root_schema_version=5,
    )
    _rewrite_release_references(
        connection,
        _json_tables(connection, _RETAINED_REFERENCE_COLUMNS),
        add_workspace_scope=True,
    )


def downgrade() -> None:
    connection = op.get_bind()
    system_release_count = connection.execute(
        sa.text("SELECT COUNT(*) FROM plugin_releases WHERE scope = 'system'")
    ).scalar_one()
    if system_release_count:
        raise RuntimeError(
            "Cannot downgrade 0019: System Plugin release data cannot be represented "
            "by the Workspace-only schema"
        )

    json_tables = _json_tables(connection, _JSON_COLUMNS)
    system_reference_locations = _system_reference_locations(connection, json_tables)
    if system_reference_locations:
        raise RuntimeError(
            "Cannot downgrade 0019: System Plugin release pins or provenance exist in "
            + ", ".join(system_reference_locations)
        )

    _rewrite_release_references(
        connection,
        _json_tables(connection, _GRAPH_DOCUMENT_COLUMNS),
        add_workspace_scope=False,
        root_schema_version=4,
    )
    _rewrite_release_references(
        connection,
        _json_tables(connection, _RETAINED_REFERENCE_COLUMNS),
        add_workspace_scope=False,
    )

    for index_name in (
        _WORKSPACE_DESCRIPTOR_INDEX,
        _SYSTEM_DESCRIPTOR_INDEX,
        _WORKSPACE_REVISION_INDEX,
        _SYSTEM_REVISION_INDEX,
    ):
        op.drop_index(index_name, table_name="plugin_releases")

    with op.batch_alter_table("plugin_releases") as batch:
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_scope_policy",
            type_="check",
        )
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_execution_policy",
            type_="check",
        )
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_scope_workspace",
            type_="check",
        )
        batch.drop_constraint(
            "ck_plugin_releases_plugin_release_scope",
            type_="check",
        )
        batch.drop_constraint("pk_plugin_releases", type_="primary")
        batch.alter_column(
            "workspace_id",
            existing_type=sa.Uuid(),
            nullable=False,
        )
        batch.create_primary_key(
            "pk_plugin_releases",
            ["workspace_id", "slug", "revision"],
        )
        batch.create_unique_constraint(
            "uq_plugin_releases_workspace_slug_descriptor",
            ("workspace_id", "slug", "descriptor_digest"),
        )
        batch.drop_column("distribution")
        batch.drop_column("execution_policy")
        batch.drop_column("scope")
        batch.drop_column("id")

    op.create_index(
        "ix_plugin_releases_workspace_slug_revision",
        "plugin_releases",
        ["workspace_id", "slug", "revision"],
        unique=False,
    )
