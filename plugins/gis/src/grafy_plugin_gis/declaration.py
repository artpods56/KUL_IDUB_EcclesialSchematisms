from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.plugins import Plugin
from grafy_core.table_contracts import TABLE_DATA


GIS = Plugin(
    slug="external.gis",
    title="GIS",
    capabilities=(
        PluginRuntimeCapability.NATIVE_GDAL,
        PluginRuntimeCapability.NETWORK_EGRESS,
        PluginRuntimeCapability.STAGED_UPLOADS,
    ),
)
GIS.register_artifact_type_dependency(TABLE_DATA)
