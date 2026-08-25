import tomllib
from pathlib import Path

from grafy_core.artifact_contracts import RASTER_IMAGE, TEXT_VALUE
from grafy_core.domain.plugin_releases import PluginCatalogManifest
from grafy_core.plugins import PluginRegistry
from grafy_core.prompt_contracts import PROMPT_MESSAGE
from grafy_plugin_prompt import PROMPTS


def test_prompt_plugin_preserves_catalog_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(PROMPTS)
    registry.freeze()
    manifest = PluginCatalogManifest.from_plugin(PROMPTS)

    assert PROMPTS.slug == "builtin.prompt"
    assert PROMPTS.artifact_type_dependencies == (TEXT_VALUE, RASTER_IMAGE)
    assert {artifact.key for artifact in PROMPTS.artifact_types} == {
        PROMPT_MESSAGE.key
    }
    assert {(node.operator_id, node.operator_version) for node in manifest.nodes} == {
        ("prompt.message.create", 2),
    }


def test_prompt_package_has_exact_vendored_sdk_supply() -> None:
    project = Path(__file__).resolve().parents[1]
    document = tomllib.loads((project / "pyproject.toml").read_text())

    assert document["project"]["name"] == "grafy-plugin-prompt"
    assert "grafy-core==0.1.0" in document["project"]["dependencies"]
    assert document.get("tool", {}).get("uv", {}).get("sources", {}) == {}
    assert 'source = { registry = "wheels" }' in (project / "uv.lock").read_text()
    assert (project / "wheels/grafy_core-0.1.0-py3-none-any.whl").is_file()
