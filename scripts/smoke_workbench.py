import asyncio
import json
from pathlib import Path

from PIL import Image, ImageDraw

from notarius_core.prototype import (
    ArtifactRefSequence,
    ArtifactWriterRegistry,
    FakeOcrEngine,
    ImageSequenceMergeNode,
    InMemoryUnitOfWork,
    InputMaterializer,
    LocalUploadImageSourceNode,
    NodeExecutionContext,
    NodeRuntime,
    OCR_PAGE_RESULT,
    OcrPageResultOutputWriter,
    OutputPersister,
    PersistedNodeOutput,
    PilImageResolver,
    ResolverRegistry,
    SOURCE_PAGE_IMAGE,
    SourcePageImageOutputWriter,
    TesseractOcrNode,
)
from notarius_storage import LocalFileObjectStore


WORKSPACE = Path(".notarius-artifacts/prototype-manual-run").resolve()
UPLOADS = WORKSPACE / "uploads"
OBJECT_STORE = WORKSPACE / "objects"
BUCKET = "prototype-artifacts"


async def main() -> None:
    image_paths = create_sample_images(UPLOADS)

    uow = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(OBJECT_STORE)
    resolver_registry = ResolverRegistry(
        [
            PilImageResolver(
                uow=uow,
                storage=storage,
            )
        ]
    )
    runtime = NodeRuntime(
        materializer=InputMaterializer(resolver_registry),
        persister=OutputPersister(
            ArtifactWriterRegistry(
                [
                    SourcePageImageOutputWriter(
                        storage=storage,
                        uow=uow,
                        bucket=BUCKET,
                    ),
                    OcrPageResultOutputWriter(uow=uow, engine="fake"),
                ]
            )
        ),
    )

    source_output = await runtime.bind(
        LocalUploadImageSourceNode(staging_root=UPLOADS),
        NodeExecutionContext(node_id="local_upload_1"),
    )(
        {},
        config={
            "connector_id": "local_upload",
            "selection": [
                {
                    "connector_id": "local_upload",
                    "external_uri": path.as_uri(),
                    "display_name": path.name,
                    "size_bytes": path.stat().st_size,
                    "order_index": index,
                }
                for index, path in enumerate(image_paths)
            ],
        },
    )
    source_pages = output_sequence(source_output, "pages")

    merge_output = await runtime.bind(
        ImageSequenceMergeNode(),
        NodeExecutionContext(node_id="merge_1"),
    )({"sequences": [source_pages]})
    merged_pages = output_sequence(merge_output, "pages")

    first_image = await resolver_registry.resolve(
        merged_pages.item_refs[0],
        Image.Image,
    )

    ocr_output = await runtime.bind(
        TesseractOcrNode(FakeOcrEngine()),
        NodeExecutionContext(node_id="ocr_1"),
    )({"pages": merged_pages})
    ocr_pages = output_sequence(ocr_output, "results")

    async with uow as entered:
        source_artifacts = await entered.artifacts.list_by_type(SOURCE_PAGE_IMAGE.key)
        ocr_artifacts = await entered.artifacts.list_by_type(OCR_PAGE_RESULT.key)

    if len(ocr_artifacts) == 0 or ocr_artifacts[0].inline_payload is None:
        raise RuntimeError("OCR writer did not persist an inline payload")

    result = {
        "workspace": str(WORKSPACE),
        "source_sequence_id": source_pages.sequence_id,
        "merged_sequence_id": merged_pages.sequence_id,
        "ocr_sequence_id": ocr_pages.sequence_id,
        "source_artifact_count": len(source_artifacts),
        "ocr_artifact_count": len(ocr_artifacts),
        "first_artifact_ref": merged_pages.item_refs[0].model_dump(mode="json"),
        "first_ocr_ref": ocr_pages.item_refs[0].model_dump(mode="json"),
        "first_image_size": list(first_image.size),
        "first_ocr_text": ocr_artifacts[0].inline_payload["text"],
        "segments": merged_pages.metadata["segments"],
    }
    print(json.dumps(result, indent=2, default=str))


def output_sequence(output: object, name: str) -> ArtifactRefSequence:
    if not isinstance(output, PersistedNodeOutput):
        raise RuntimeError(f"Node output is not persisted for {name!r}")
    value = output[name]
    if not isinstance(value, ArtifactRefSequence):
        raise RuntimeError(f"Output {name!r} is not an ArtifactRefSequence")
    return value


def create_sample_images(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / "page-001.png", directory / "page-002.png"]
    for index, path in enumerate(paths):
        image = Image.new("RGB", (260, 90), color="white")
        draw = ImageDraw.Draw(image)
        draw.text((20, 30), f"PAGE {index + 1}", fill="black")
        image.save(path, format="PNG")
    return paths


if __name__ == "__main__":
    asyncio.run(main())
