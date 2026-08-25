from grafy_core.domain.plugin_capabilities import PluginRuntimeCapability
from grafy_core.plugins import Plugin
from grafy_core.artifact_contracts import RASTER_IMAGE


OCR = Plugin(
    slug="external.ocr",
    title="OCR",
    capabilities=(
        PluginRuntimeCapability.NATIVE_TESSERACT,
    ),
)
OCR.register_artifact_type_dependency(RASTER_IMAGE)
