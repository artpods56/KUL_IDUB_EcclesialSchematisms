from importlib.metadata import requires

from grafy_core.plugins import Plugin, PluginRegistry
from grafy_plugin_sql.plugin import SQL


def test_manifest_loader_target_preserves_system_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(SQL)
    registry.freeze()

    assert isinstance(SQL, Plugin)
    assert SQL.slug == "external.sql"
    assert {registration.key for registration in SQL.nodes} == {
        ("sql.statement.raw", 1),
        ("sql.artifacts.query", 1),
        ("sql.postgresql.execute", 1),
    }
    assert "grafy-core==0.1.0" in (requires("grafy-plugin-sql") or [])
