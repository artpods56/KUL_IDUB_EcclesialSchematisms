from hashlib import sha256
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from typing import Annotated, Literal, final, override

from pydantic import BaseModel, ConfigDict, Field, StrictBytes, StrictInt, StrictStr

from notarius_core.artifacts import (
    ArtifactObject,
    ArtifactRef,
    ArtifactTypeKey,
    ArtifactTypeSpec,
    JsonObject,
    NodeConfig,
    NodeInput,
    NodeOutput,
    UnitOfWorkPort,
)
from notarius_core.nodes import Node, NodeExecutionContext, OutPort
from notarius_core.plugins import Plugin
from notarius_core.ports.staged_uploads import StagedUploadUnitOfWorkPort
from notarius_core.ports.storage import FileMetadata, FileStoragePort, SaveFileCommand
from notarius_core.runtime.persistence import (
    ArtifactOutputWriter,
    ArtifactWriteContext,
)
from notarius_core.staged_upload_paths import resolve_persisted_staged_upload_path


RasterImageContentType = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
    "image/bmp",
]


class RasterImageContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: StrictBytes = Field(min_length=1)
    content_type: RasterImageContentType
    filename: StrictStr | None = Field(default=None, min_length=1)


RASTER_IMAGE = ArtifactTypeSpec(
    key=ArtifactTypeKey("image.raster", 1),
    title="Raster image",
)


IMAGES = Plugin(
    slug="builtin.image",
    title="Image",
)
IMAGES.register_artifact_type(RASTER_IMAGE)


@final
class RasterImageOutputWriter(ArtifactOutputWriter):
    artifact_type = RASTER_IMAGE.key

    def __init__(
        self,
        *,
        storage: FileStoragePort,
        uow: UnitOfWorkPort,
        bucket: str,
        storage_backend: str = "local",
    ) -> None:
        self._storage = storage
        self._uow = uow
        self._bucket = bucket
        self._storage_backend = storage_backend

    @override
    async def write(
        self,
        value: object,
        context: ArtifactWriteContext,
    ) -> ArtifactRef:
        image = RasterImageContent.model_validate(value)
        content_hash = sha256(image.content).hexdigest()
        storage_path = (
            f"workspaces/{context.node_context.workspace_id}/"
            f"{self.artifact_type.id}/v{self.artifact_type.schema_version}/"
            f"{content_hash}{self._suffix_for(image.content_type)}"
        )
        file_metadata: FileMetadata = {
            "original_filename": image.filename,
            "artifact_kind": self.artifact_type.id,
            "sha256": content_hash,
        }
        if context.node_context.node_id is not None:
            file_metadata["job_id"] = context.node_context.node_id

        try:
            stored_file = await self._storage.save(
                SaveFileCommand(
                    bucket=self._bucket,
                    path=storage_path,
                    stream=BytesIO(image.content),
                    content_type=image.content_type,
                    metadata=file_metadata,
                    allow_overwrite=True,
                )
            )
        except Exception as exc:
            node_id = context.node_context.node_id or "<unknown>"
            raise RuntimeError(
                f"Failed to persist raster image output for node {node_id!r} "
                f"at {self._bucket}/{storage_path}"
            ) from exc

        provenance: JsonObject = {
            input_name: [
                {
                    "artifact_id": str(ref.artifact_id),
                    "artifact_type": ref.artifact_type,
                    "schema_version": ref.schema_version,
                }
                for ref in refs
            ]
            for input_name, refs in context.provenance.refs_by_input.items()
        }
        metadata: JsonObject = {
            "producer_node_id": context.node_context.node_id,
            "original_filename": image.filename,
            "content_hash": content_hash,
            "storage_byte_size": stored_file.byte_size,
            "storage_sha256": stored_file.sha256,
        }
        if provenance:
            metadata["provenance"] = provenance
        metadata.update(context.metadata)
        artifact = ArtifactObject(
            workspace_id=context.node_context.workspace_id,
            artifact_type=self.artifact_type.id,
            schema_version=self.artifact_type.schema_version,
            content_type=image.content_type,
            storage_backend=self._storage_backend,
            bucket=stored_file.bucket,
            object_key=stored_file.path,
            byte_size=stored_file.byte_size,
            sha256=stored_file.sha256,
            metadata=metadata,
        )
        async with self._uow as uow:
            await uow.artifacts.add(artifact)
            await uow.commit()
        return artifact.ref()

    def _suffix_for(self, content_type: RasterImageContentType) -> str:
        suffixes: dict[RasterImageContentType, str] = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/tiff": ".tiff",
            "image/bmp": ".bmp",
        }
        return suffixes[content_type]


IMAGES.register_writer(
    lambda context: RasterImageOutputWriter(
        storage=context.storage,
        uow=context.uow,
        bucket=context.bucket,
        storage_backend=context.storage_backend,
    )
)


class ImageUploadError(RuntimeError):
    pass


class ImageUploadItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_key: StrictStr = Field(min_length=1)
    filename: StrictStr = Field(min_length=1)
    byte_size: StrictInt = Field(ge=0)


class ImageUploadConfig(NodeConfig):
    uploads: list[ImageUploadItem] = Field(
        min_length=1,
        description="Staged image uploads in output order.",
    )


class ImageUploadInput(NodeInput):
    pass


class ImageUploadOutput(NodeOutput):
    images: Annotated[
        list[RasterImageContent],
        OutPort(RASTER_IMAGE),
        Field(description="Ordered raster images imported from staged uploads."),
    ]


@IMAGES.node(
    operator_id="image.upload",
    version=1,
    title="Upload images",
    factory=lambda context: UploadImagesNode(
        uploads_dir=context.uploads_dir,
        unit_of_work=context.uow,
    ),
)
@final
class UploadImagesNode(Node[ImageUploadConfig, ImageUploadInput, ImageUploadOutput]):
    """Imports staged image uploads as an ordered raster image sequence."""

    def __init__(
        self,
        uploads_dir: Path,
        unit_of_work: StagedUploadUnitOfWorkPort,
    ) -> None:
        self._uploads_dir = uploads_dir.expanduser().resolve()
        self._unit_of_work = unit_of_work

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        config: ImageUploadConfig,
        _inputs: ImageUploadInput,
        /,
    ) -> ImageUploadOutput:
        images: list[RasterImageContent] = []
        for upload in config.uploads:
            try:
                path = await resolve_persisted_staged_upload_path(
                    self._uploads_dir,
                    self._unit_of_work,
                    workspace_id=context.workspace_id,
                    upload_key=upload.upload_key,
                )
            except (ValueError, FileNotFoundError) as exc:
                raise ImageUploadError(str(exc)) from exc
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise ImageUploadError(
                    f"Failed to read staged image upload {upload.upload_key!r} "
                    f"from {path}"
                ) from exc

            if len(content) != upload.byte_size:
                raise ImageUploadError(
                    f"Staged image upload {upload.upload_key!r} changed size: "
                    f"expected {upload.byte_size}, got {len(content)}"
                )
            images.append(
                RasterImageContent(
                    content=content,
                    content_type=self._content_type_for(upload),
                    filename=upload.filename,
                )
            )
        return ImageUploadOutput(images=images)

    def _content_type_for(self, upload: ImageUploadItem) -> RasterImageContentType:
        content_type = guess_type(upload.filename)[0]
        if content_type in (
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/tiff",
            "image/bmp",
        ):
            return content_type
        raise ImageUploadError(
            f"Unsupported image content type for upload {upload.upload_key!r} "
            f"with filename {upload.filename!r}"
        )
