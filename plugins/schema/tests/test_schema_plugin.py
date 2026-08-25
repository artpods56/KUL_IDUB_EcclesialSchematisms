import tomllib
from pathlib import Path

from grafy_core.domain.plugin_releases import PluginCatalogManifest
from grafy_core.plugins import PluginRegistry
from grafy_core.schema_contracts import JSON_SCHEMA
from grafy_plugin_schema import SCHEMAS


def test_schema_plugin_preserves_catalog_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(SCHEMAS)
    registry.freeze()
    manifest = PluginCatalogManifest.from_plugin(SCHEMAS)

    assert SCHEMAS.slug == "builtin.schema"
    assert {artifact.key for artifact in SCHEMAS.artifact_types} == {
        JSON_SCHEMA.key
    }
    assert {(node.operator_id, node.operator_version) for node in manifest.nodes} == {
        ("schema.builder", 1),
    }


def test_schema_package_has_exact_vendored_sdk_supply() -> None:
    project = Path(__file__).resolve().parents[1]
    document = tomllib.loads((project / "pyproject.toml").read_text())

    assert document["project"]["name"] == "grafy-plugin-schema"
    assert "grafy-core==0.1.0" in document["project"]["dependencies"]
    assert document.get("tool", {}).get("uv", {}).get("sources", {}) == {}
    assert 'source = { registry = "wheels" }' in (project / "uv.lock").read_text()
    assert (project / "wheels/grafy_core-0.1.0-py3-none-any.whl").is_file()
