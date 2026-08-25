from grafy_core.artifact_contracts import RASTER_IMAGE, TEXT_VALUE
from grafy_core.plugins import Plugin


PROMPTS = Plugin(
    slug="builtin.prompt",
    title="Prompt",
)
PROMPTS.register_artifact_type_dependency(TEXT_VALUE)
PROMPTS.register_artifact_type_dependency(RASTER_IMAGE)
