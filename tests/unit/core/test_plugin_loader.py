from hashlib import sha256

from pydantic import ValidationError
import pytest

from grafy_core.runtime.plugin_loader import (
    PluginGuestLoaderManifest,
)


def test_loader_manifest_is_canonical_and_content_addressed() -> None:
    manifest = PluginGuestLoaderManifest(
        slug="notes",
        loader_target="grafy_plugin_notes.plugin:NOTES",
    )

    payload = manifest.canonical_json_bytes()

    assert payload.endswith(b"\n")
    assert PluginGuestLoaderManifest.from_json_bytes(payload) == manifest
    assert manifest.digest == sha256(payload).hexdigest()


def test_loader_manifest_is_scope_neutral_and_accepts_project_owned_targets() -> None:
    manifest = PluginGuestLoaderManifest(
        slug="notes",
        loader_target="grafy_plugin_notes.plugin:NOTES",
    )

    assert manifest.loader_target == "grafy_plugin_notes.plugin:NOTES"


@pytest.mark.parametrize(
    "loader_target",
    (
        "grafy_plugin_notes",
        "grafy_plugin_notes.plugin.NOTES",
        "grafy_plugin_notes.plugin:NOTES.extra",
        "../grafy_plugin_notes:NOTES",
    ),
)
def test_loader_manifest_rejects_non_import_targets(loader_target: str) -> None:
    with pytest.raises(ValidationError):
        PluginGuestLoaderManifest(
            slug="notes",
            loader_target=loader_target,
        )
