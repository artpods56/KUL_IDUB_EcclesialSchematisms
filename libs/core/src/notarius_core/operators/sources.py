from hashlib import sha256
from mimetypes import guess_type
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Protocol, final, override
from urllib.parse import unquote, urlparse

from pydantic import BaseModel, Field, model_validator

from notarius_core.artifacts import (
    SOURCE_PAGE_IMAGE,
    ArtifactRef,
    ArtifactRefSequence,
    JsonObject,
    NoConfig,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)
from notarius_core.plugins import Plugin


SOURCES = Plugin(
    slug="builtin.sources",
    title="Sources",
)
SOURCES.register_artifact_type(SOURCE_PAGE_IMAGE)


ImageContentType = Literal[
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
    "image/bmp",
]


class SourceConnectorError(RuntimeError):
    pass


class SourceSelectionError(RuntimeError):
    pass


class SourceSelectionItem(BaseModel):
    connector_id: str
    external_uri: str
    display_name: str
    order_index: int = Field(ge=0)
    size_bytes: int | None = Field(default=None, ge=0)
    content_type: ImageContentType | None = None
    version_token: str | None = None


class SourceImageContent(BaseModel):
    selection: SourceSelectionItem
    content: bytes
    content_type: ImageContentType
    byte_size: int = Field(ge=0)
    sha256: str

    @model_validator(mode="after")
    def validate_byte_size(self) -> "SourceImageContent":
        if self.byte_size != len(self.content):
            message = (
                f"Source image byte_size does not match content length: expected "
                f"{self.byte_size}, got {len(self.content)}"
            )
            raise ValueError(message)
        return self


class ImageSourceAdapter(Protocol):
    connector_id: str

    async def fetch(self, selection: SourceSelectionItem) -> SourceImageContent: ...


@final
class LocalUploadImageAdapter(ImageSourceAdapter):
    connector_id = "local_upload"

    def __init__(self, staging_root: Path | None = None) -> None:
        self._staging_root = staging_root.resolve() if staging_root is not None else None

    @override
    async def fetch(self, selection: SourceSelectionItem) -> SourceImageContent:
        if selection.connector_id != self.connector_id:
            message = (
                f"Local upload adapter received selection for connector "
                f"{selection.connector_id!r}"
            )
            raise SourceConnectorError(message)

        path = self._path_from_uri(selection.external_uri)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise SourceConnectorError(
                f"Failed to read local upload image {path}"
            ) from exc

        byte_size = len(content)
        if selection.size_bytes is not None and selection.size_bytes != byte_size:
            message = (
                f"Local upload image size changed for {selection.external_uri!r}: "
                f"expected {selection.size_bytes}, got {byte_size}"
            )
            raise SourceConnectorError(message)

        content_type = selection.content_type or self._content_type_for(path)
        return SourceImageContent(
            selection=selection,
            content=content,
            content_type=content_type,
            byte_size=byte_size,
            sha256=sha256(content).hexdigest(),
        )

    def _path_from_uri(self, external_uri: str) -> Path:
        parsed = urlparse(external_uri)
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path))
        elif parsed.scheme == "":
            path = Path(external_uri)
        else:
            raise SourceConnectorError(
                f"Local upload adapter cannot read URI {external_uri!r}"
            )

        resolved = path.expanduser().resolve()
        if self._staging_root is not None and not resolved.is_relative_to(
            self._staging_root
        ):
            message = f"Local upload image {resolved} is outside staging root {self._staging_root}"
            raise SourceConnectorError(message)
        return resolved

    def _content_type_for(self, path: Path) -> ImageContentType:
        guessed_type = guess_type(path.name)[0]
        if guessed_type in (
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/tiff",
            "image/bmp",
        ):
            return guessed_type
        raise SourceConnectorError(
            f"Unsupported local upload image content type for {path}"
        )


class ImageSourceConfig(NodeConfig):
    connector_id: str
    selection: list[SourceSelectionItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> "ImageSourceConfig":
        seen_order_indexes: set[int] = set()
        for item in self.selection:
            if item.connector_id != self.connector_id:
                message = (
                    f"Source selection connector mismatch: expected "
                    f"{self.connector_id!r}, got {item.connector_id!r}"
                )
                raise ValueError(message)
            if item.order_index in seen_order_indexes:
                raise ValueError(
                    f"Duplicate source selection order index {item.order_index}"
                )
            seen_order_indexes.add(item.order_index)
        return self


class ImageSourceInput(NodeInput):
    pass


class ImageSourceOutput(NodeOutput):
    pages: Annotated[
        list[SourceImageContent],
        OutPort(produces=SOURCE_PAGE_IMAGE),
        Field(description="Ordered images imported from the selected source."),
    ]


class ImageSourceNode(Node[ImageSourceConfig, ImageSourceInput, ImageSourceOutput]):
    operator_id: ClassVar[str] = "source.image.import"
    operator_version: ClassVar[int] = 1

    def __init__(
        self,
        adapter: ImageSourceAdapter,
    ) -> None:
        self._adapter: ImageSourceAdapter = adapter

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: ImageSourceConfig,
        _inputs: ImageSourceInput,
        /,
    ) -> ImageSourceOutput:
        if config.connector_id != self._adapter.connector_id:
            message = (
                f"Image source node configured for {config.connector_id!r}, "
                f"but adapter is {self._adapter.connector_id!r}"
            )
            raise SourceSelectionError(message)

        pages: list[SourceImageContent] = []
        for selection in sorted(config.selection, key=lambda item: item.order_index):
            image = await self._adapter.fetch(selection)
            pages.append(image)

        return ImageSourceOutput(pages=pages)


@SOURCES.node(
    operator_id="source.local_upload.images",
    version=1,
    title="Local upload image source",
    factory=lambda context: LocalUploadImageSourceNode(
        staging_root=context.uploads_dir
    ),
)
@final
class LocalUploadImageSourceNode(ImageSourceNode):
    """Imports staged local images as an ordered artifact sequence."""

    def __init__(
        self,
        staging_root: Path | None = None,
    ) -> None:
        super().__init__(
            adapter=LocalUploadImageAdapter(staging_root=staging_root),
        )


class ImageSequenceMergeInput(NodeInput):
    sequences: Annotated[
        list[ArtifactRefSequence],
        InPort(accepts=SOURCE_PAGE_IMAGE, variadic=True),
        Field(
            min_length=1,
            description="Image sequences to concatenate in connection order.",
        ),
    ]

    @model_validator(mode="after")
    def validate_sequences(self) -> "ImageSequenceMergeInput":
        for sequence in self.sequences:
            if sequence.artifact_type != SOURCE_PAGE_IMAGE.key.id:
                message = (
                    f"ImageSequenceMerge expected {SOURCE_PAGE_IMAGE.key.id}, "
                    f"got {sequence.artifact_type}"
                )
                raise ValueError(message)
            if sequence.schema_version != SOURCE_PAGE_IMAGE.key.schema_version:
                message = (
                    f"ImageSequenceMerge expected schema version "
                    f"{SOURCE_PAGE_IMAGE.key.schema_version}, got {sequence.schema_version}"
                )
                raise ValueError(message)
        return self


class ImageSequenceMergeOutput(NodeOutput):
    pages: Annotated[
        ArtifactRefSequence,
        OutPort(produces=SOURCE_PAGE_IMAGE),
        Field(description="One ordered sequence containing every input image."),
    ]


@SOURCES.node(
    operator_id="source.image_sequence.merge",
    version=1,
    title="Image sequence merge",
)
class ImageSequenceMergeNode(
    Node[NoConfig, ImageSequenceMergeInput, ImageSequenceMergeOutput]
):
    """Concatenates image sequences without copying their artifact payloads."""

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ImageSequenceMergeInput,
        /,
    ) -> ImageSequenceMergeOutput:
        item_refs: list[ArtifactRef] = []
        segments: list[JsonObject] = []
        start_index = 0
        for input_index, sequence in enumerate(inputs.sequences):
            count = len(sequence.item_refs)
            source_node_id = sequence.metadata.get("source_node_id")
            source_id = (
                source_node_id
                if isinstance(source_node_id, str) and source_node_id
                else f"input_{input_index}"
            )
            segments.append(
                {
                    "source_node_id": source_id,
                    "source_sequence_id": sequence.sequence_id,
                    "start_index": start_index,
                    "count": count,
                }
            )
            item_refs.extend(sequence.item_refs)
            start_index += count

        metadata: JsonObject = {
            "source_node_id": context.node_id,
            "segments": segments,
        }
        return ImageSequenceMergeOutput(
            pages=ArtifactRefSequence.from_key(
                key=SOURCE_PAGE_IMAGE.key,
                item_refs=item_refs,
                metadata=metadata,
            )
        )
