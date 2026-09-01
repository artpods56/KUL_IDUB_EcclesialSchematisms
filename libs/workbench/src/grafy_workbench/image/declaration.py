from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.plugins import Plugin


IMAGES = Plugin(
    slug="image",
    title="Image",
    capabilities=(PluginRuntimeCapability.STAGED_UPLOADS,),
)
