import tomllib
from pathlib import Path

from grafy_core.domain.plugin_releases import PluginCatalogManifest
from grafy_core.plugins import PluginRegistry
from grafy_core.table_contracts import TABLE_DATA
from grafy_plugin_table import TABLES


def test_table_plugin_preserves_catalog_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(TABLES)
    registry.freeze()
    manifest = PluginCatalogManifest.from_plugin(TABLES)

    assert TABLES.slug == "builtin.table"
    assert {artifact.key for artifact in TABLES.artifact_types} == {TABLE_DATA.key}
    assert {(node.operator_id, node.operator_version) for node in manifest.nodes} == {
        ("table.file.import", 1),
        ("table.fuzzy_match", 1),
        ("table.text.normalize", 1),
    }


def test_table_package_has_exact_vendored_sdk_supply() -> None:
    project = Path(__file__).resolve().parents[1]
    document = tomllib.loads((project / "pyproject.toml").read_text())

    assert document["project"]["name"] == "grafy-plugin-table"
    assert "grafy-core==0.1.0" in document["project"]["dependencies"]
    assert document.get("tool", {}).get("uv", {}).get("sources", {}) == {}
    assert (project / "wheels/grafy_core-0.1.0-py3-none-any.whl").is_file()
