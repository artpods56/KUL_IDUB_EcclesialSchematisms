from importlib.metadata import requires

from grafy_core.plugins import Plugin, PluginRegistry
from grafy_plugin_ocr.plugin import OCR


def test_manifest_loader_target_preserves_system_identity_and_freezes() -> None:
    registry = PluginRegistry()
    registry.install(OCR)
    registry.freeze()

    assert isinstance(OCR, Plugin)
    assert OCR.slug == "external.ocr"
    assert {registration.key for registration in OCR.nodes} == {
        ("ocr.tesseract.pages", 2),
    }
    assert "grafy-core==0.1.0" in (requires("grafy-plugin-ocr") or [])
