"""Emit a Plugin catalog contract from an isolated publisher subprocess.

The inspector runs inside the Plugin's own locked environment against an
unpacked source snapshot. It imports the fixed ``grafy_plugin`` package and
reads its ``PLUGIN`` declaration; no module or object names are supplied by
the caller.
"""

from importlib import import_module

from pydantic import BaseModel, ConfigDict

from grafy_core.domain.plugin_releases import (
    PluginCapabilityManifest,
    PluginCatalogManifest,
)
from grafy_core.plugins import Plugin


class InspectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    catalog: PluginCatalogManifest
    capabilities: PluginCapabilityManifest


def inspect_plugin() -> InspectionResult:
    module = import_module("grafy_plugin")
    plugin = getattr(module, "PLUGIN", None)
    if not isinstance(plugin, Plugin):
        raise SystemExit(
            "The installed project must export a grafy_core Plugin named "
            "'PLUGIN' from the 'grafy_plugin' package"
        )
    return InspectionResult(
        catalog=PluginCatalogManifest.from_plugin(plugin),
        capabilities=PluginCapabilityManifest(capabilities=plugin.capabilities),
    )


def main() -> None:
    print(inspect_plugin().model_dump_json())


if __name__ == "__main__":
    main()
