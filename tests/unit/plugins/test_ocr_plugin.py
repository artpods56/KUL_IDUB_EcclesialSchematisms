from pathlib import Path

from notarius_core.artifacts import InMemoryUnitOfWork
from notarius_core.plugins import PluginRegistry, PluginRuntimeContext
from notarius_plugin_ocr import OCR
from notarius_plugin_ocr.artifacts import (
    MISTRAL_OCR_RESPONSE,
    OCR_PAGE_RESULT,
)
from notarius_storage import LocalFileObjectStore


def test_ocr_plugin_declares_complete_runtime_contributions(tmp_path: Path) -> None:
    registry = PluginRegistry()
    registry.install(OCR)
    context = PluginRuntimeContext(
        workspace=tmp_path,
        uploads_dir=tmp_path / "uploads",
        storage=LocalFileObjectStore(tmp_path / "objects"),
        uow=InMemoryUnitOfWork(),
        bucket="artifacts",
    )

    assert OCR.slug == "external.ocr"
    assert OCR.title == "OCR"
    assert {registration.key for registration in registry.nodes} == {
        ("ocr.tesseract.pages", 1),
        ("ocr.mistral.tables", 1),
        ("table.markdown.extract", 1),
    }
    assert {
        registration.node_class.plugin_slug for registration in registry.nodes
    } == {"external.ocr"}
    assert {
        registration.node_class.title for registration in registry.nodes
    } == {"Tesseract OCR", "Mistral OCR 4", "Extract Markdown Tables"}
    assert [artifact_type.key for artifact_type in registry.artifact_types] == [
        OCR_PAGE_RESULT.key,
        MISTRAL_OCR_RESPONSE.key,
    ]
    assert [
        registry.build_node(*registration.key, context).__class__
        for registration in registry.nodes
    ] == [registration.node_class for registration in registry.nodes]

    resolvers = registry.build_resolvers(context)
    writers = registry.build_writers(context)

    assert len(resolvers) == 3
    assert len(writers) == 2
    assert {writer.artifact_type for writer in writers} == {
        OCR_PAGE_RESULT.key,
        MISTRAL_OCR_RESPONSE.key,
    }
