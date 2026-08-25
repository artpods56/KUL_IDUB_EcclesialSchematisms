from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.plugins import Plugin


TABLES = Plugin(
    slug="builtin.table",
    title="Table",
    capabilities=(PluginRuntimeCapability.STAGED_UPLOADS,),
)


__all__ = ["TABLES"]
