import tomllib
from pathlib import Path

from grafy_core.artifact_contracts import INTEGER_VALUE
from grafy_core.domain.plugin_releases import PluginCatalogManifest
from grafy_core.plugins import PluginRegistry
from grafy_plugin_arithmetic import ARITHMETIC


def test_arithmetic_plugin_preserves_catalog_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(ARITHMETIC)
    registry.freeze()
    manifest = PluginCatalogManifest.from_plugin(ARITHMETIC)

    assert ARITHMETIC.slug == "builtin.arithmetic"
    assert {artifact.key for artifact in ARITHMETIC.artifact_types} == {
        INTEGER_VALUE.key
    }
    assert {(node.operator_id, node.operator_version) for node in manifest.nodes} == {
        ("arithmetic.number", 1),
        ("arithmetic.integer_sequence", 1),
        ("arithmetic.add", 1),
        ("arithmetic.subtract", 1),
        ("arithmetic.multiply", 1),
        ("arithmetic.sum", 1),
    }


def test_arithmetic_package_has_exact_vendored_sdk_supply() -> None:
    project = Path(__file__).resolve().parents[1]
    document = tomllib.loads((project / "pyproject.toml").read_text())

    assert document["project"]["name"] == "grafy-plugin-arithmetic"
    assert "grafy-core==0.1.0" in document["project"]["dependencies"]
    assert (project / "wheels/grafy_core-0.1.0-py3-none-any.whl").is_file()
