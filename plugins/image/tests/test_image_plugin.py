import tomllib
from pathlib import Path

from grafy_core.artifact_contracts import RASTER_IMAGE
from grafy_core.domain.plugin_releases import PluginCatalogManifest
from grafy_core.plugins import PluginRegistry
from grafy_plugin_image.plugin import IMAGES


def test_image_plugin_preserves_catalog_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(IMAGES)
    registry.freeze()
    manifest = PluginCatalogManifest.from_plugin(IMAGES)

    assert IMAGES.slug == "builtin.image"
    assert {artifact.key for artifact in IMAGES.artifact_types} == {RASTER_IMAGE.key}
    assert RASTER_IMAGE.bundle.format == "binary-file"
    assert RASTER_IMAGE.bundle.version == 1
    assert {(node.operator_id, node.operator_version) for node in manifest.nodes} == {
        ("image.upload", 1),
    }


def test_image_package_has_exact_vendored_sdk_supply() -> None:
    project = Path(__file__).resolve().parents[1]
    document = tomllib.loads((project / "pyproject.toml").read_text())

    assert document["project"]["name"] == "grafy-plugin-image"
    assert "grafy-core==0.1.0" in document["project"]["dependencies"]
    assert "entry-points" not in document["project"]
    assert (project / "wheels/grafy_core-0.1.0-py3-none-any.whl").is_file()
