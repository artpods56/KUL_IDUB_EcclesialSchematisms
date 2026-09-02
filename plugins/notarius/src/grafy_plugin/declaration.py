from grafy_core.artifact_contracts import RASTER_IMAGE, TEXT_VALUE
from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.plugins import Plugin
from grafy_core.schema_contracts import JSON_SCHEMA
from grafy_core.table_contracts import TABLE_DATA


PLUGIN = Plugin(
    slug="external.notarius",
    title="Notarius",
    capabilities=(
        PluginRuntimeCapability.NETWORK_EGRESS,
        PluginRuntimeCapability.NODE_SECRETS,
    ),
)
PLUGIN.register_artifact_type_dependency(RASTER_IMAGE)
PLUGIN.register_artifact_type_dependency(TEXT_VALUE)
PLUGIN.register_artifact_type_dependency(JSON_SCHEMA)
PLUGIN.register_artifact_type_dependency(TABLE_DATA)
