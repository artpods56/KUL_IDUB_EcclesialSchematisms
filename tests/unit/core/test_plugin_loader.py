from hashlib import sha256

from pydantic import ValidationError
import pytest

from grafy_core.domain.plugin_identity import PluginReleaseScope
from grafy_core.runtime.plugin_loader import (
    PluginGuestLoaderManifest,
    WORKSPACE_PLUGIN_LOADER_TARGET,
)


def test_loader_manifest_is_canonical_and_content_addressed() -> None:
    manifest = PluginGuestLoaderManifest(
        scope=PluginReleaseScope.SYSTEM,
        slug="builtin.text",
        loader_target="grafy_plugin_text.plugin:TEXT",
    )

    payload = manifest.canonical_json_bytes()

    assert payload.endswith(b"\n")
    assert PluginGuestLoaderManifest.from_json_bytes(payload) == manifest
    assert manifest.digest == sha256(payload).hexdigest()


def test_workspace_loader_manifest_requires_the_fixed_package_contract() -> None:
    manifest = PluginGuestLoaderManifest(
        scope=PluginReleaseScope.WORKSPACE,
        slug="notes",
        loader_target=WORKSPACE_PLUGIN_LOADER_TARGET,
    )

    assert manifest.loader_target == "grafy_plugin:PLUGIN"
    with pytest.raises(ValidationError, match="must use grafy_plugin:PLUGIN"):
        PluginGuestLoaderManifest(
            scope=PluginReleaseScope.WORKSPACE,
            slug="notes",
            loader_target="grafy_plugin_notes:NOTES",
        )


@pytest.mark.parametrize(
    "loader_target",
    (
        "grafy_plugin_text",
        "grafy_plugin_text.plugin.TEXT",
        "grafy_plugin_text.plugin:TEXT.extra",
        "../grafy_plugin_text:TEXT",
    ),
)
def test_loader_manifest_rejects_non_import_targets(loader_target: str) -> None:
    with pytest.raises(ValidationError):
        PluginGuestLoaderManifest(
            scope=PluginReleaseScope.SYSTEM,
            slug="builtin.text",
            loader_target=loader_target,
        )
