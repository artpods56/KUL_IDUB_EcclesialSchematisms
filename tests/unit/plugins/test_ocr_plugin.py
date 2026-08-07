import tomllib
from pathlib import Path

from notarius_core.artifacts import InMemoryUnitOfWork
from notarius_core.operators.images import IMAGES, RASTER_IMAGE
from notarius_core.plugins import PluginOrigin, PluginRegistry, PluginRuntimeContext
from notarius_plugin_ocr import OCR
from notarius_plugin_ocr.artifacts import (
    MISTRAL_OCR_RESPONSE,
    OCR_PAGE_RESULT,
    TABLE_FRAGMENT,
)
from notarius_plugin_ocr.resolvers import EncodedPageImageResolver, PilImageResolver
from notarius_storage import LocalFileObjectStore


def test_ocr_plugin_declares_complete_runtime_contributions(tmp_path: Path) -> None:
    registry = PluginRegistry()
    registry.install(IMAGES, origin=PluginOrigin.BUILTIN)
    registry.install(OCR, origin=PluginOrigin.EXTERNAL)
    context = PluginRuntimeContext(
        workspace=tmp_path,
        uploads_dir=tmp_path / "uploads",
        storage=LocalFileObjectStore(tmp_path / "objects"),
        uow=InMemoryUnitOfWork(),
        bucket="artifacts",
    )

    assert OCR.slug == "external.ocr"
    assert OCR.title == "OCR"
    registry.freeze()

    assert [plugin.origin for plugin in registry.plugins] == [
        PluginOrigin.BUILTIN,
        PluginOrigin.EXTERNAL,
    ]
    ocr_registrations = [
        registration
        for registration in registry.nodes
        if registration.node_class.plugin_slug == OCR.slug
    ]
    assert {registration.key for registration in ocr_registrations} == {
        ("ocr.tesseract.pages", 2),
        ("ocr.mistral.tables", 2),
        ("table.markdown.extract", 1),
    }
    assert {registration.node_class.plugin_slug for registration in ocr_registrations} == {
        "external.ocr"
    }
    assert {registration.node_class.title for registration in ocr_registrations} == {
        "Tesseract OCR",
        "Mistral OCR 4",
        "Extract Markdown Tables",
    }
    assert [artifact_type.key for artifact_type in registry.artifact_types] == [
        RASTER_IMAGE.key,
        OCR_PAGE_RESULT.key,
        MISTRAL_OCR_RESPONSE.key,
        TABLE_FRAGMENT.key,
    ]
    assert [
        registry.build_node(*registration.key, context).__class__
        for registration in ocr_registrations
    ] == [registration.node_class for registration in ocr_registrations]

    resolvers = registry.build_resolvers(context)
    writers = registry.build_writers(context)

    assert len(resolvers) == 3
    assert {
        resolver.source
        for resolver in resolvers
        if isinstance(resolver, (EncodedPageImageResolver, PilImageResolver))
    } == {RASTER_IMAGE.key}
    assert len(writers) == 4
    assert {writer.artifact_type for writer in writers} == {
        RASTER_IMAGE.key,
        OCR_PAGE_RESULT.key,
        MISTRAL_OCR_RESPONSE.key,
        TABLE_FRAGMENT.key,
    }


def test_ocr_package_metadata_declares_plugin_entry_point() -> None:
    project_root = Path(__file__).parents[3]
    metadata = tomllib.loads(
        (project_root / "plugins" / "ocr" / "pyproject.toml").read_text()
    )

    assert metadata["project"]["name"] == "notarius-plugin-ocr"
    assert metadata["project"]["entry-points"]["notarius.plugins"] == {
        "ocr": "notarius_plugin_ocr.plugin:OCR"
    }
