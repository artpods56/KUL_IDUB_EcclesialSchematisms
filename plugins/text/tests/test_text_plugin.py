import tomllib
from pathlib import Path

from grafy_core.artifact_contracts import TEXT_VALUE
from grafy_core.domain.plugin_releases import PluginCatalogManifest
from grafy_core.plugins import PluginRegistry
from grafy_plugin_text import TEXT


def test_text_plugin_preserves_catalog_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(TEXT)
    registry.freeze()
    manifest = PluginCatalogManifest.from_plugin(TEXT)

    assert TEXT.slug == "builtin.text"
    assert TEXT.artifact_type_dependencies == ()
    assert TEXT_VALUE.key in {artifact.key for artifact in TEXT.artifact_types}
    assert {(node.operator_id, node.operator_version) for node in manifest.nodes} == {
        ("text.input", 1),
        ("text.as_markdown", 1),
        ("text.split", 1),
        ("text.replace", 1),
        ("text.join", 1),
    }


def test_text_package_has_exact_vendored_sdk_supply() -> None:
    project = Path(__file__).resolve().parents[1]
    document = tomllib.loads((project / "pyproject.toml").read_text())

    assert document["project"]["name"] == "grafy-plugin-text"
    assert "grafy-core==0.1.0" in document["project"]["dependencies"]
    assert (project / "wheels/grafy_core-0.1.0-py3-none-any.whl").is_file()
