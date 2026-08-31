from grafy_core.artifact_contracts import RASTER_IMAGE
from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.plugins import Plugin
from grafy_core.prompt_contracts import PROMPT_MESSAGE
from grafy_core.schema_contracts import JSON_SCHEMA


LLM = Plugin(
    slug="external.llm",
    title="LLM",
    capabilities=(
        PluginRuntimeCapability.NETWORK_EGRESS,
        PluginRuntimeCapability.NODE_SECRETS,
    ),
)
LLM.register_artifact_type_dependency(PROMPT_MESSAGE)
LLM.register_artifact_type_dependency(RASTER_IMAGE)
LLM.register_artifact_type_dependency(JSON_SCHEMA)
