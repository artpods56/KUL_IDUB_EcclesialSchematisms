import asyncio
import json
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageDraw

from grafy_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    InMemoryUnitOfWork,
)
from grafy_core.nodes import NodeExecutionContext
from grafy_core.operators.arithmetic import (
    INTEGER_VALUE,
    IntegerValueOutputWriter,
    IntegerValuePayload,
)
from grafy_core.operators.images import IMAGES, RASTER_IMAGE
from grafy_core.operators.sequences import (
    CollectNode,
    CountNode,
    ItemAtNode,
    SliceNode,
)
from grafy_core.plugins import PluginOrigin, PluginRegistry, PluginRuntimeContext
from grafy_core.runtime.execution import NodeRuntime, PersistedNodeOutput
from grafy_core.runtime.materialization import InputMaterializer
from grafy_core.runtime.persistence import (
    ArtifactWriterRegistry,
    OutputPersister,
)
from grafy_core.runtime.resolvers import ResolverRegistry
from grafy_plugin_ocr import OCR
from grafy_plugin_ocr.artifacts import OCR_PAGE_RESULT
from grafy_storage import LocalFileObjectStore


WORKSPACE = Path(".grafy-artifacts/workbench-smoke").resolve()
WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000901")
UPLOADS = WORKSPACE / "uploads"
OBJECT_STORE = WORKSPACE / "objects"
BUCKET = "workbench-artifacts"


async def main() -> None:
    image_paths = create_sample_images(UPLOADS / str(WORKSPACE_ID))

    uow = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(OBJECT_STORE)
    plugin_registry = PluginRegistry()
    plugin_registry.install(IMAGES, origin=PluginOrigin.BUILTIN)
    plugin_registry.install(OCR, origin=PluginOrigin.EXTERNAL)
    plugin_registry.freeze()
    plugin_context = PluginRuntimeContext(
        workspace=WORKSPACE,
        uploads_dir=UPLOADS,
        storage=storage,
        uow=uow,
        bucket=BUCKET,
    )
    resolver_registry = ResolverRegistry(
        list(plugin_registry.build_resolvers(plugin_context))
    )
    runtime = NodeRuntime(
        materializer=InputMaterializer(resolver_registry),
        persister=OutputPersister(
            ArtifactWriterRegistry(
                [
                    IntegerValueOutputWriter(uow=uow),
                    *plugin_registry.build_writers(plugin_context),
                ]
            )
        ),
    )

    upload_output = await runtime.bind(
        plugin_registry.build_node("image.upload", 1, plugin_context),
        NodeExecutionContext(workspace_id=WORKSPACE_ID, node_id="image_upload_1"),
    )(
        {},
        config={
            "uploads": [
                {
                    "upload_key": path.name,
                    "filename": path.name,
                    "byte_size": path.stat().st_size,
                }
                for path in image_paths
            ],
        },
    )
    uploaded_images = output_sequence(upload_output, "images")

    collect_output = await runtime.bind(
        CollectNode(),
        NodeExecutionContext(workspace_id=WORKSPACE_ID, node_id="collect_1"),
        artifact_type_bindings={"T": RASTER_IMAGE.key},
    )({"items": [uploaded_images]})
    collected_images = output_sequence(collect_output, "items")

    count_output = await runtime.bind(
        CountNode(),
        NodeExecutionContext(workspace_id=WORKSPACE_ID, node_id="count_1"),
        artifact_type_bindings={"T": RASTER_IMAGE.key},
    )({"items": collected_images})
    count_ref = output_ref(count_output, "count")

    slice_output = await runtime.bind(
        SliceNode(),
        NodeExecutionContext(workspace_id=WORKSPACE_ID, node_id="slice_1"),
        artifact_type_bindings={"T": RASTER_IMAGE.key},
    )(
        {"items": collected_images},
        config={"start": 0, "count": 1},
    )
    selected_pages = output_sequence(slice_output, "items")

    pick_output = await runtime.bind(
        ItemAtNode(),
        NodeExecutionContext(workspace_id=WORKSPACE_ID, node_id="pick_1"),
        artifact_type_bindings={"T": RASTER_IMAGE.key},
    )(
        {"items": collected_images},
        config={"index": 0},
    )
    first_image_ref = output_ref(pick_output, "item")
    first_image = await resolver_registry.resolve(
        first_image_ref,
        Image.Image,
        WORKSPACE_ID,
    )

    ocr_output = await runtime.bind(
        plugin_registry.build_node("ocr.tesseract.pages", 2, plugin_context),
        NodeExecutionContext(workspace_id=WORKSPACE_ID, node_id="ocr_1"),
    )({"pages": selected_pages})
    ocr_pages = output_sequence(ocr_output, "results")

    async with uow as entered:
        image_artifacts = await entered.artifacts.list_by_type(
            WORKSPACE_ID,
            RASTER_IMAGE.key,
        )
        integer_artifacts = await entered.artifacts.list_by_type(
            WORKSPACE_ID,
            INTEGER_VALUE.key,
        )
        ocr_artifacts = await entered.artifacts.list_by_type(
            WORKSPACE_ID,
            OCR_PAGE_RESULT.key,
        )

    if len(integer_artifacts) != 1:
        raise RuntimeError("Count did not persist exactly one integer artifact")
    count_payload = integer_artifacts[0].inline_payload
    if count_payload is None:
        raise RuntimeError("Count did not persist an inline payload")
    count_value = IntegerValuePayload.model_validate(count_payload).value
    if count_value != len(collected_images.item_refs):
        raise RuntimeError("Count did not persist the collected sequence length")
    if len(ocr_artifacts) == 0:
        raise RuntimeError("OCR writer did not persist an artifact")
    ocr_payload = ocr_artifacts[0].inline_payload
    if ocr_payload is None:
        raise RuntimeError("OCR writer did not persist an inline payload")

    result: dict[str, object] = {
        "workspace": str(WORKSPACE),
        "uploaded_sequence_id": uploaded_images.sequence_id,
        "collected_sequence_id": collected_images.sequence_id,
        "selected_sequence_id": selected_pages.sequence_id,
        "ocr_sequence_id": ocr_pages.sequence_id,
        "raster_image_count": len(image_artifacts),
        "sequence_item_count": count_value,
        "ocr_artifact_count": len(ocr_artifacts),
        "count_artifact_ref": count_ref.model_dump(mode="json"),
        "first_artifact_ref": first_image_ref.model_dump(mode="json"),
        "first_ocr_ref": ocr_pages.item_refs[0].model_dump(mode="json"),
        "first_image_size": list(first_image.size),
        "first_ocr_text": ocr_payload["text"],
        "segments": collected_images.metadata["collect_segments"],
    }
    print(json.dumps(result, indent=2, default=str))


def output_sequence(output: object, name: str) -> ArtifactRefSequence:
    if not isinstance(output, PersistedNodeOutput):
        raise RuntimeError(f"Node output is not persisted for {name!r}")
    value = output[name]
    if not isinstance(value, ArtifactRefSequence):
        raise RuntimeError(f"Output {name!r} is not an ArtifactRefSequence")
    return value


def output_ref(output: object, name: str) -> ArtifactRef:
    if not isinstance(output, PersistedNodeOutput):
        raise RuntimeError(f"Node output is not persisted for {name!r}")
    value = output[name]
    if not isinstance(value, ArtifactRef):
        raise RuntimeError(f"Output {name!r} is not an ArtifactRef")
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
