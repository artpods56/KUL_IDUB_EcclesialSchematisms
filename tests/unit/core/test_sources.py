from pathlib import Path
from uuid import uuid4

import pytest
from PIL import Image

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    InMemoryUnitOfWork,
    SOURCE_PAGE_IMAGE,
)
from notarius_core.nodes import NodeExecutionContext
from notarius_core.operators.sources import (
    ImageSequenceMergeNode,
    LocalUploadImageSourceNode,
)
from notarius_core.runtime.execution import NodeRuntime
from notarius_core.runtime.materialization import InputMaterializer
from notarius_core.runtime.persistence import (
    ArtifactWriterRegistry,
    OutputPersister,
    PersistedNodeOutput,
    SourcePageImageOutputWriter,
)
from notarius_core.runtime.resolvers import ResolverRegistry
from notarius_plugin_ocr.artifacts import OCR_PAGE_RESULT
from notarius_plugin_ocr.persistence import OcrPageResultOutputWriter
from notarius_plugin_ocr.resolvers import PilImageResolver
from notarius_plugin_ocr.tesseract import FakeOcrEngine, TesseractOcrNode
from notarius_storage import LocalFileObjectStore


def write_png(path: Path, size: tuple[int, int]) -> None:
    image = Image.new("RGB", size, color="white")
    image.save(path, format="PNG")


@pytest.mark.asyncio
async def test_runtime_chains_source_merge_and_ocr_writers(tmp_path: Path) -> None:
    staging_root = tmp_path / "uploads"
    staging_root.mkdir()
    first = staging_root / "page-001.png"
    second = staging_root / "page-002.png"
    write_png(first, (3, 2))
    write_png(second, (5, 4))

    uow = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(tmp_path / "object-store")
    runtime = NodeRuntime(
        materializer=InputMaterializer(
            ResolverRegistry([PilImageResolver(uow=uow, storage=storage)])
        ),
        persister=OutputPersister(
            ArtifactWriterRegistry(
                [
                    SourcePageImageOutputWriter(
                        storage=storage,
                        uow=uow,
                        bucket="artifacts",
                    ),
                    OcrPageResultOutputWriter(uow=uow, engine="fake"),
                ]
            )
        ),
    )

    source_output = await runtime.bind(
        LocalUploadImageSourceNode(staging_root=staging_root),
        NodeExecutionContext(node_id="local_upload_1"),
    )(
        {},
        config={
            "connector_id": "local_upload",
            "selection": [
                {
                    "connector_id": "local_upload",
                    "external_uri": second.as_uri(),
                    "display_name": "page-002.png",
                    "size_bytes": second.stat().st_size,
                    "content_type": "image/png",
                    "order_index": 1,
                },
                {
                    "connector_id": "local_upload",
                    "external_uri": first.as_uri(),
                    "display_name": "page-001.png",
                    "size_bytes": first.stat().st_size,
                    "content_type": "image/png",
                    "order_index": 0,
                },
            ],
        },
    )
    assert isinstance(source_output, PersistedNodeOutput)
    source_pages = source_output["pages"]

    assert isinstance(source_pages, ArtifactRefSequence)
    assert source_pages.artifact_type == SOURCE_PAGE_IMAGE.key.id
    assert len(source_pages.item_refs) == 2

    merge_output = await runtime.bind(
        ImageSequenceMergeNode(),
        NodeExecutionContext(node_id="merge_1"),
    )({"sequences": [source_pages]})
    assert isinstance(merge_output, PersistedNodeOutput)
    merge_pages = merge_output["pages"]

    assert isinstance(merge_pages, ArtifactRefSequence)
    assert merge_pages.item_refs == source_pages.item_refs
    assert merge_pages.metadata["segments"] == [
        {
            "source_node_id": "input_0",
            "source_sequence_id": source_pages.sequence_id,
            "start_index": 0,
            "count": 2,
        }
    ]

    ocr_output = await runtime.bind(
        TesseractOcrNode(FakeOcrEngine()),
        NodeExecutionContext(node_id="ocr_1"),
    )({"pages": merge_pages})
    assert isinstance(ocr_output, PersistedNodeOutput)
    ocr_results = ocr_output["results"]

    assert isinstance(ocr_results, ArtifactRefSequence)
    assert ocr_results.artifact_type == OCR_PAGE_RESULT.key.id
    assert len(ocr_results.item_refs) == 2

    async with uow as entered:
        source_artifacts = await entered.artifacts.list_by_type(SOURCE_PAGE_IMAGE.key)
        ocr_artifacts = await entered.artifacts.list_by_type(OCR_PAGE_RESULT.key)

    assert [artifact.metadata["original_filename"] for artifact in source_artifacts] == [
        "page-001.png",
        "page-002.png",
    ]
    assert all(artifact.bucket == "artifacts" for artifact in source_artifacts)
    assert all(artifact.object_key is not None for artifact in source_artifacts)
    texts: list[object] = []
    for artifact in ocr_artifacts:
        assert artifact.inline_payload is not None
        texts.append(artifact.inline_payload["text"])

    assert texts == ["fake OCR image 3x2", "fake OCR image 5x4"]


@pytest.mark.asyncio
async def test_ocr_many_input_requires_artifact_ref_sequence() -> None:
    page_ref = ArtifactRef.from_key(
        artifact_id=uuid4(),
        key=SOURCE_PAGE_IMAGE.key,
    )
    materializer = InputMaterializer(ResolverRegistry())

    with pytest.raises(RuntimeError, match="wrap refs in an ArtifactRefSequence"):
        await materializer.materialize(
            TesseractOcrNode.input_contract,
            {"pages": [page_ref]},
        )
