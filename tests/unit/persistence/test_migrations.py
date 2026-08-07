import json
import importlib
from pathlib import Path
from uuid import UUID

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy import Column, Integer, MetaData, Table, create_engine, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects import sqlite

from notarius_api.settings import get_settings
from notarius_persistence.schema import staged_uploads


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_tenant_rebuild_uses_postgresql_temporary_constraint_names() -> None:
    migration = importlib.import_module(
        "infra.db.migrations.versions.0008_tenant_existing_resources"
    )
    mock_postgresql = type("Dialect", (), {"name": "postgresql"})()
    temporary_name = migration._rebuild_constraint_name(
        type("Connection", (), {"dialect": mock_postgresql})(),
        "pk_saved_graphs",
        "u",
    )
    assert temporary_name == "tmp_0008u_pk_saved_graphs"
    table = Table(
        "_0008_saved_graphs",
        MetaData(),
        Column("id", Integer, primary_key=True),
    )
    table.primary_key.name = temporary_name
    ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    assert "CONSTRAINT tmp_0008u_pk_saved_graphs PRIMARY KEY" in ddl
    sqlite_upload_ddl = str(
        CreateTable(staged_uploads).compile(dialect=sqlite.dialect())
    )
    postgres_upload_ddl = str(
        CreateTable(staged_uploads).compile(dialect=postgresql.dialect())
    )
    assert "instr(upload_key, char(92)) = 0" in sqlite_upload_ddl
    assert "position(chr(92) in upload_key) = 0" in postgres_upload_ddl


def test_tenant_upgrade_preflight_leaves_no_temporary_tables_and_retries_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "preflight" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(config, "0007_identity_workspace_foundation")
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(
            text("UPDATE workspaces SET slug = 'temporarily-invalid' WHERE id = :id"),
            {"id": "00000000000000000000000000000007"},
        )

    with pytest.raises(RuntimeError, match="deterministic local workspace"):
        command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert not any(
            table_name.startswith("_0008")
            for table_name in inspect(connection).get_table_names()
        )
        connection.commit()
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(
            text("UPDATE workspaces SET slug = 'local' WHERE id = :id"),
            {"id": "00000000000000000000000000000007"},
        )
    command.upgrade(config, "head")
    get_settings.cache_clear()


def test_tenant_downgrade_preflight_rejects_leftovers_before_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "downgrade-preflight" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(text("CREATE TABLE _0008d_saved_graphs (id INTEGER)"))

    with pytest.raises(RuntimeError, match="temporary table"):
        command.downgrade(config, "0007_identity_workspace_foundation")
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        assert "_0008d_saved_graphs" in inspect(connection).get_table_names()
        connection.execute(text("DROP TABLE _0008d_saved_graphs"))
    command.downgrade(config, "0007_identity_workspace_foundation")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert not any(
            table_name.startswith("_0008d_")
            for table_name in inspect(connection).get_table_names()
        )
    get_settings.cache_clear()


def test_alembic_migration_upgrades_downgrades_and_has_no_schema_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "fresh" / "nested" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()

    config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(config, "0003_node_secrets")

    graph_id = UUID("00000000-0000-0000-0000-000000000401")
    document: dict[str, object] = {
        "schema_version": 3,
        "nodes": [],
        "edges": [],
    }
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO saved_graphs "
                "(id, name, document, revision, created_at, updated_at) "
                "VALUES (:id, :name, :document, :revision, :created_at, :updated_at)"
            ),
            {
                "id": graph_id.hex,
                "name": "Existing graph",
                "document": json.dumps(document),
                "revision": 7,
                "created_at": "2026-07-14 08:00:00",
                "updated_at": "2026-07-16 09:30:00",
            },
        )

    command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        row = (
            connection.execute(
                text(
                    "SELECT graph_id, revision, name, document, created_at "
                    "FROM saved_graph_revisions"
                )
            )
            .mappings()
            .one()
        )
        assert row["graph_id"] == graph_id.hex
        assert row["revision"] == 7
        assert row["name"] == "Existing graph"
        assert json.loads(row["document"]) == document
        assert str(row["created_at"]) == "2026-07-16 09:30:00"

    command.downgrade(config, "0003_node_secrets")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert "saved_graph_revisions" not in inspect(connection).get_table_names()

    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")

    command.upgrade(config, "head")
    assert database_path.exists()
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert set(inspect(connection).get_table_names()) == {
            "alembic_version",
            "artifact_objects",
            "graph_execution_node_results",
            "graph_execution_requested_nodes",
            "graph_executions",
            "invocation_cache_entries",
            "materialized_node_outputs",
            "node_secrets",
            "users",
            "oidc_identities",
            "oidc_login_transactions",
            "oidc_bootstrap_owner_mappings",
            "workspaces",
            "workspace_memberships",
            "auth_sessions",
            "personal_access_tokens",
            "security_audit_events",
            "saved_graphs",
            "saved_graph_revisions",
            "staged_uploads",
        }
    command.check(config)

    command.downgrade(config, "base")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert inspect(connection).get_table_names() == ["alembic_version"]

    command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        assert set(inspect(connection).get_table_names()) == {
            "alembic_version",
            "artifact_objects",
            "graph_execution_node_results",
            "graph_execution_requested_nodes",
            "graph_executions",
            "invocation_cache_entries",
            "materialized_node_outputs",
            "node_secrets",
            "users",
            "oidc_identities",
            "oidc_login_transactions",
            "oidc_bootstrap_owner_mappings",
            "workspaces",
            "workspace_memberships",
            "auth_sessions",
            "personal_access_tokens",
            "security_audit_events",
            "saved_graphs",
            "saved_graph_revisions",
            "staged_uploads",
        }

    get_settings.cache_clear()


def test_identity_migration_creates_sealed_local_workspace_and_audit_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "identity" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")

    command.upgrade(config, "0007_identity_workspace_foundation")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        local = (
            connection.execute(
                text(
                    "SELECT slug, kind, personal_owner_user_id "
                    "FROM workspaces WHERE slug = 'local'"
                )
            )
            .mappings()
            .one()
        )
        assert local == {
            "slug": "local",
            "kind": "shared",
            "personal_owner_user_id": None,
        }
        assert connection.execute(text("SELECT COUNT(*) FROM users")).scalar_one() == 0
        workspaces_ddl = connection.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'workspaces'"
            )
        ).scalar_one()
        assert "GLOB" not in workspaces_ddl.upper()
        assert "lower(trim(slug))" in workspaces_ddl
        indexes = {
            row[1]
            for row in connection.execute(
                text("PRAGMA index_list('security_audit_events')")
            )
        }
        assert indexes >= {
            "ix_security_audit_events_workspace_occurred_at",
            "ix_security_audit_events_actor_occurred_at",
            "ix_security_audit_events_operation_occurred_at",
            "ix_security_audit_events_retention",
        }
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO workspaces "
                    "(id, slug, name, kind, personal_owner_user_id, "
                    "created_at, updated_at) VALUES "
                    "(:id, :slug, :name, :kind, NULL, :created_at, :updated_at)"
                ),
                {
                    "id": "00000000000000000000000000000008",
                    "slug": "Not-Normalized",
                    "name": "Invalid",
                    "kind": "shared",
                    "created_at": "2026-08-07 00:00:00",
                    "updated_at": "2026-08-07 00:00:00",
                },
            )

    get_settings.cache_clear()


def test_saved_graph_revision_migration_backfills_the_current_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "backfill" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()

    config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(config, "0003_node_secrets")

    graph_id = UUID("00000000-0000-0000-0000-000000000401")
    document: dict[str, object] = {
        "schema_version": 3,
        "nodes": [],
        "edges": [],
    }
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO saved_graphs "
                "(id, name, document, revision, created_at, updated_at) "
                "VALUES (:id, :name, :document, :revision, :created_at, :updated_at)"
            ),
            {
                "id": graph_id.hex,
                "name": "Existing graph",
                "document": json.dumps(document),
                "revision": 7,
                "created_at": "2026-07-14 08:00:00",
                "updated_at": "2026-07-16 09:30:00",
            },
        )

    command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        row = (
            connection.execute(
                text(
                    "SELECT graph_id, revision, name, document, created_at "
                    "FROM saved_graph_revisions"
                )
            )
            .mappings()
            .one()
        )
        assert row["graph_id"] == graph_id.hex
        assert row["revision"] == 7
        assert row["name"] == "Existing graph"
        assert json.loads(row["document"]) == document
        assert str(row["created_at"]) == "2026-07-16 09:30:00"

    command.downgrade(config, "0003_node_secrets")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        assert "saved_graph_revisions" not in inspect(connection).get_table_names()

    get_settings.cache_clear()


def test_tenant_migration_backfills_all_0006_resources_and_checks_composite_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "tenant-backfill" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(config, "0006_execution_history")

    graph_id = UUID("00000000-0000-0000-0000-000000000801")
    execution_id = UUID("00000000-0000-0000-0000-000000000802")
    artifact_id = UUID("00000000-0000-0000-0000-000000000803")
    generation_id = UUID("00000000-0000-0000-0000-000000000804")
    workflow_run_id = UUID("00000000-0000-0000-0000-000000000805")
    timestamp = "2026-08-07 08:00:00"
    document = json.dumps({"schema_version": 3, "nodes": [], "edges": []})
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO saved_graphs "
                "(id, name, document, revision, created_at, updated_at) "
                "VALUES (:id, 'Migrated graph', :document, 1, :created_at, :updated_at)"
            ),
            {
                "id": graph_id.hex,
                "document": document,
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO saved_graph_revisions "
                "(graph_id, revision, name, document, created_at) "
                "VALUES (:graph_id, 1, 'Migrated graph', :document, :created_at)"
            ),
            {"graph_id": graph_id.hex, "document": document, "created_at": timestamp},
        )
        connection.execute(
            text(
                "INSERT INTO artifact_objects "
                "(id, artifact_type, schema_version, content_type, storage_backend, "
                "metadata) VALUES (:id, 'test.artifact', 1, 'application/json', "
                "'inline', '{}')"
            ),
            {"id": artifact_id.hex},
        )
        connection.execute(
            text(
                "INSERT INTO invocation_cache_entries "
                "(key_sha256, generation, outputs, created_at) "
                "VALUES (:key, :generation, '[]', :created_at)"
            ),
            {
                "key": "a" * 64,
                "generation": generation_id.hex,
                "created_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO materialized_node_outputs "
                "(graph_id, graph_revision, node_id, workflow_run_id, outputs, "
                "materialized_at) VALUES (:graph_id, 1, 'node', :workflow_run_id, "
                "'[]', :created_at)"
            ),
            {
                "graph_id": graph_id.hex,
                "workflow_run_id": workflow_run_id.hex,
                "created_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO node_secrets "
                "(graph_id, node_id, name, operator_id, operator_version, key_id, "
                "dependency_sha256, nonce, ciphertext, created_at, updated_at) "
                "VALUES (:graph_id, 'node', 'secret', 'test.operator', 1, 'key', "
                ":dependency, :nonce, :ciphertext, :created_at, :updated_at)"
            ),
            {
                "graph_id": graph_id.hex,
                "dependency": "b" * 64,
                "nonce": b"0" * 12,
                "ciphertext": b"ciphertext",
                "created_at": timestamp,
                "updated_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO graph_executions "
                "(execution_id, graph_id, graph_revision, status, scope, created_at) "
                "VALUES (:execution_id, :graph_id, 1, 'succeeded', 'all', :created_at)"
            ),
            {
                "execution_id": execution_id.hex,
                "graph_id": graph_id.hex,
                "created_at": timestamp,
            },
        )
        connection.execute(
            text(
                "INSERT INTO graph_execution_requested_nodes "
                "(execution_id, node_id, position) VALUES (:execution_id, 'node', 0)"
            ),
            {"execution_id": execution_id.hex},
        )
        connection.execute(
            text(
                "INSERT INTO graph_execution_node_results "
                "(execution_id, node_id, position, status, outputs, artifact_count, "
                "completed_at) VALUES (:execution_id, 'node', 0, 'succeeded', '[]', "
                "0, :completed_at)"
            ),
            {"execution_id": execution_id.hex, "completed_at": timestamp},
        )

    command.upgrade(config, "head")
    with create_engine(f"sqlite:///{database_path}").connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        for table_name in (
            "saved_graphs",
            "saved_graph_revisions",
            "artifact_objects",
            "invocation_cache_entries",
            "materialized_node_outputs",
            "node_secrets",
            "graph_executions",
            "graph_execution_requested_nodes",
            "graph_execution_node_results",
        ):
            assert (
                connection.execute(
                    text(
                        f"SELECT COUNT(*) FROM {table_name} "
                        "WHERE workspace_id = '00000000000000000000000000000007'"
                    )
                ).scalar_one()
                == 1
            )
        assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
        connection.execute(
            text(
                "INSERT INTO workspaces "
                "(id, slug, name, kind, created_at, updated_at) VALUES "
                "(:id, 'other', 'Other', 'shared', :created_at, :created_at)"
            ),
            {"id": "00000000000000000000000000000009", "created_at": timestamp},
        )
        with pytest.raises(IntegrityError):
            connection.execute(
                text(
                    "INSERT INTO saved_graph_revisions "
                    "(workspace_id, graph_id, revision, name, document, created_at) "
                    "VALUES (:workspace_id, :graph_id, 2, 'Foreign', :document, "
                    ":created_at)"
                ),
                {
                    "workspace_id": "00000000000000000000000000000009",
                    "graph_id": graph_id.hex,
                    "document": document,
                    "created_at": timestamp,
                },
            )

    get_settings.cache_clear()


def test_direct_0007_downgrade_refuses_identity_data_but_allows_empty_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "identity-guard" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{database_path}",
    )
    get_settings.cache_clear()
    config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(config, "0007_identity_workspace_foundation")
    with create_engine(f"sqlite:///{database_path}").begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users "
                "(id, active, created_at, updated_at) VALUES "
                "('00000000000000000000000000000008', 1, :created_at, :created_at)"
            ),
            {"created_at": "2026-08-07 08:00:00"},
        )
    with pytest.raises(RuntimeError, match="identity/security data"):
        command.downgrade(config, "0006_execution_history")

    empty_database_path = tmp_path / "identity-empty" / "migrated.sqlite3"
    monkeypatch.setenv(
        "NOTARIUS_DATABASE_URL",
        f"sqlite+aiosqlite:///{empty_database_path}",
    )
    get_settings.cache_clear()
    empty_config = Config(REPOSITORY_ROOT / "alembic.ini")
    command.upgrade(empty_config, "0007_identity_workspace_foundation")
    command.downgrade(empty_config, "0006_execution_history")
    get_settings.cache_clear()
