from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.plugins import Plugin
from grafy_core.table_contracts import TABLE_DATA


SQL = Plugin(
    slug="external.sql",
    title="SQL",
    capabilities=(
        PluginRuntimeCapability.NODE_SECRETS,
        PluginRuntimeCapability.POSTGRESQL_EGRESS,
        PluginRuntimeCapability.UNTRUSTED_SQL,
    ),
)
SQL.register_artifact_type_dependency(TABLE_DATA)
