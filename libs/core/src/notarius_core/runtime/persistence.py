import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
from typing import Any, Protocol, cast, final, override
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from pydantic import BaseModel

from notarius_core.ports.storage import FileMetadata, FileStoragePort, SaveFileCommand
from notarius_core.artifacts import (
    SOURCE_PAGE_IMAGE,
    TABLE_CSV_BUNDLE,
    ArtifactObject,
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    JsonObject,
    UnitOfWorkPort,
)
from notarius_core.nodes import (
    NodeExecutionContext,
    OutputContract,
    OutputPortSpec,
    PortShape,
)
from notarius_core.operators.sources import SourceImageContent
from notarius_core.operators.tables import TableCsvBundle
from notarius_core.runtime.materialization import MaterializationProvenance


@dataclass(frozen=True, slots=True)
class ArtifactWriteContext:
    node_context: NodeExecutionContext
    provenance: MaterializationProvenance
    item_index: int | None = None
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PersistedNodeOutput:
    values: dict[str, object]

    def __getitem__(self, name: str) -> object:
        return self.values[name]

    def __getattr__(self, name: str) -> object:
        try:
            return self.values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ArtifactOutputWriter(Protocol):
    artifact_type: ArtifactTypeKey

    async def write(
        self,
        value: object,
        context: ArtifactWriteContext,
    ) -> ArtifactRef: ...


class ArtifactWriterRegistry:
    def __init__(self, writers: list[ArtifactOutputWriter] | None = None) -> None:
        self._writers: dict[ArtifactTypeKey, ArtifactOutputWriter] = {}
        for writer in writers or []:
            self.register(writer)

    def register(self, writer: ArtifactOutputWriter) -> None:
        if writer.artifact_type in self._writers:
            artifact_type = writer.artifact_type
            raise ValueError(
                f"Output writer already registered for {artifact_type.id}@"
                f"{artifact_type.schema_version}"
            )
        self._writers[writer.artifact_type] = writer

    def writer_for(self, artifact_type: ArtifactTypeKey) -> ArtifactOutputWriter:
        writer = self._writers.get(artifact_type)
        if writer is None:
            message = (
                f"No output writer registered for {artifact_type.id}@"
                f"{artifact_type.schema_version}"
            )
            raise RuntimeError(message)
        return writer


@final
class InlineModelOutputWriter[T: BaseModel](ArtifactOutputWriter):
    """Persists a typed Pydantic payload as an inline JSON artifact."""

    def __init__(
        self,
        *,
        artifact_type: ArtifactTypeKey,
        model: type[T],
        uow: UnitOfWorkPort,
    ) -> None:
        self.artifact_type = artifact_type
        self._model = model
        self._uow = uow

    @override
    async def write(
        self,
        value: object,
        context: ArtifactWriteContext,
    ) -> ArtifactRef:
        payload = self._model.model_validate(value)
        payload_json = cast(JsonObject, payload.model_dump(mode="json", by_alias=True))
        payload_bytes = json.dumps(
            payload_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        provenance: dict[str, object] = {
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
        }
        if provenance:
            metadata["provenance"] = provenance
        metadata.update(context.metadata)
        artifact = ArtifactObject(
            artifact_type=self.artifact_type.id,
            schema_version=self.artifact_type.schema_version,
            content_type="application/json",
            storage_backend="inline",
            inline_payload=payload_json,
            byte_size=len(payload_bytes),
            sha256=sha256(payload_bytes).hexdigest(),
            metadata=metadata,
        )
        async with self._uow as uow:
            await uow.artifacts.add(artifact)
            await uow.commit()
        return artifact.ref()


@final
class TableCsvBundleOutputWriter(ArtifactOutputWriter):
    artifact_type = TABLE_CSV_BUNDLE.key

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
        bundle = TableCsvBundle.model_validate(value)
        archive_stream = BytesIO()
        with ZipFile(
            archive_stream,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for csv_file in bundle.files:
                info = ZipInfo(csv_file.path, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, csv_file.content.encode("utf-8"))

        archive_bytes = archive_stream.getvalue()
        archive_hash = sha256(archive_bytes).hexdigest()
        download_name = "notarius-table-csv-export.zip"
        stored_file = await self._storage.save(
            SaveFileCommand(
                bucket=self._bucket,
                path=(
                    f"{self.artifact_type.id}/v{self.artifact_type.schema_version}/"
                    f"{archive_hash}.zip"
                ),
                stream=BytesIO(archive_bytes),
                content_type="application/zip",
                metadata={
                    "original_filename": download_name,
                    "artifact_kind": self.artifact_type.id,
                    "sha256": archive_hash,
                },
                allow_overwrite=True,
            )
        )
        manifest: list[JsonObject] = [
            {
                "path": csv_file.path,
                "content_type": "text/csv; charset=utf-8",
                "byte_size": len(csv_file.content.encode("utf-8")),
            }
            for csv_file in bundle.files
        ]
        artifact = ArtifactObject(
            artifact_type=self.artifact_type.id,
            schema_version=self.artifact_type.schema_version,
            content_type="application/zip",
            storage_backend=self._storage_backend,
            bucket=stored_file.bucket,
            object_key=stored_file.path,
            byte_size=stored_file.byte_size,
            sha256=stored_file.sha256,
            metadata={
                "producer_node_id": context.node_context.node_id,
                "download_name": download_name,
                "file_count": len(bundle.files),
                "files": manifest,
                "combined_written": any(
                    csv_file.path == "all_pages_combined.csv"
                    for csv_file in bundle.files
                ),
            },
        )
        async with self._uow as uow:
            await uow.artifacts.add(artifact)
            await uow.commit()
        return artifact.ref()


class OutputPersister:
    def __init__(self, writer_registry: ArtifactWriterRegistry) -> None:
        self._writer_registry = writer_registry

    async def persist(
        self,
        contract: OutputContract[Any],
        context: NodeExecutionContext,
        output: object,
        provenance: MaterializationProvenance,
    ) -> PersistedNodeOutput | BaseModel:
        validated_output = contract.model.model_validate(output)
        values = _model_values(validated_output)

        for name, spec in contract.ports.items():
            if name not in values:
                if spec.required:
                    raise RuntimeError(f"Missing required output {name!r}")
                continue

            values[name] = await self._persist_value(
                spec,
                values[name],
                ArtifactWriteContext(
                    node_context=context,
                    provenance=provenance,
                    item_index=context.invocation_index,
                ),
            )

        if len(contract.ports) == 0:
            return validated_output
        return PersistedNodeOutput(values=values)

    async def _persist_value(
        self,
        spec: OutputPortSpec,
        value: object,
        context: ArtifactWriteContext,
    ) -> object:
        if isinstance(value, ArtifactRef):
            if value.key() != spec.produces:
                raise RuntimeError(
                    f"Output {spec.name!r} expected {spec.produces.id}@"
                    f"{spec.produces.schema_version}, got {value.artifact_type}@"
                    f"{value.schema_version}"
                )
            if spec.shape is not PortShape.ONE:
                raise RuntimeError(
                    f"Output {spec.name!r} expected an ArtifactRefSequence, "
                    "got ArtifactRef"
                )
            return value
        if isinstance(value, ArtifactRefSequence):
            if (
                value.artifact_type != spec.produces.id
                or value.schema_version != spec.produces.schema_version
            ):
                raise RuntimeError(
                    f"Output {spec.name!r} expected {spec.produces.id}@"
                    f"{spec.produces.schema_version}, got {value.artifact_type}@"
                    f"{value.schema_version}"
                )
            if spec.shape is not PortShape.MANY:
                raise RuntimeError(
                    f"Output {spec.name!r} expected an ArtifactRef, got "
                    "ArtifactRefSequence"
                )
            return value

        items: list[object] | None = None
        if isinstance(value, Sequence) and not isinstance(
            value, str | bytes | bytearray
        ):
            items = list(cast(Sequence[object], value))

        if items is not None and all(isinstance(item, ArtifactRef) for item in items):
            refs = cast(list[ArtifactRef], items)
            for ref in refs:
                if ref.key() != spec.produces:
                    raise RuntimeError(
                        f"Output {spec.name!r} expected {spec.produces.id}@"
                        f"{spec.produces.schema_version}, got {ref.artifact_type}@"
                        f"{ref.schema_version}"
                    )
            if spec.shape == PortShape.MANY:
                return ArtifactRefSequence.from_key(
                    key=spec.produces,
                    item_refs=refs,
                )
            if len(refs) == 1:
                return refs[0]
            raise RuntimeError(
                f"Output {spec.name!r} expected one ArtifactRef, got {len(refs)}"
            )

        writer = self._writer_registry.writer_for(spec.produces)
        if spec.shape == PortShape.MANY:
            if items is None:
                raise RuntimeError(f"Output {spec.name!r} expected a sequence")
            item_refs = [
                await writer.write(
                    item,
                    ArtifactWriteContext(
                        node_context=context.node_context,
                        provenance=context.provenance,
                        item_index=index,
                    ),
                )
                for index, item in enumerate(items)
            ]
            return ArtifactRefSequence.from_key(
                key=spec.produces,
                item_refs=item_refs,
            )

        return await writer.write(cast(object, value), context)


@final
class SourcePageImageOutputWriter(ArtifactOutputWriter):
    artifact_type = SOURCE_PAGE_IMAGE.key

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
        source_image = SourceImageContent.model_validate(value)
        artifact_id = SOURCE_PAGE_IMAGE.key.id
        storage_path = (
            f"{artifact_id}/v{SOURCE_PAGE_IMAGE.key.schema_version}/"
            f"{source_image.selection.order_index}-{source_image.sha256}"
            f"{self._suffix_for(source_image.content_type)}"
        )
        metadata = _source_image_metadata(source_image, context)
        stored_file = await self._storage.save(
            SaveFileCommand(
                bucket=self._bucket,
                path=storage_path,
                stream=BytesIO(source_image.content),
                content_type=source_image.content_type,
                metadata=metadata,
                allow_overwrite=True,
            )
        )
        artifact = ArtifactObject(
            artifact_type=SOURCE_PAGE_IMAGE.key.id,
            schema_version=SOURCE_PAGE_IMAGE.key.schema_version,
            content_type=source_image.content_type,
            storage_backend=self._storage_backend,
            bucket=stored_file.bucket,
            object_key=stored_file.path,
            byte_size=stored_file.byte_size,
            sha256=stored_file.sha256,
            metadata={
                **_source_image_payload(source_image),
                "storage_byte_size": stored_file.byte_size,
                "storage_sha256": stored_file.sha256,
            },
        )
        async with self._uow as uow:
            await uow.artifacts.add(artifact)
            await uow.commit()
        return artifact.ref()

    def _suffix_for(self, content_type: str) -> str:
        suffixes = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/tiff": ".tiff",
            "image/bmp": ".bmp",
        }
        suffix = suffixes.get(content_type)
        if suffix is not None:
            return suffix
        raise RuntimeError(f"Unsupported image content type {content_type!r}")


def _model_values(value: BaseModel) -> dict[str, object]:
    return {name: getattr(value, name) for name in value.__class__.model_fields}


def _source_image_metadata(
    source_image: SourceImageContent,
    context: ArtifactWriteContext,
) -> FileMetadata:
    metadata: FileMetadata = {
        "original_filename": source_image.selection.display_name,
        "source": source_image.selection.connector_id,
        "artifact_kind": SOURCE_PAGE_IMAGE.key.id,
        "sha256": source_image.sha256,
    }
    if context.node_context.node_id is not None:
        metadata["job_id"] = context.node_context.node_id
    return metadata


def _source_image_payload(source_image: SourceImageContent) -> JsonObject:
    return {
        "source_connector": source_image.selection.connector_id,
        "source_uri": source_image.selection.external_uri,
        "source_version_token": source_image.selection.version_token,
        "original_filename": source_image.selection.display_name,
        "content_hash": source_image.sha256,
        "order_index": source_image.selection.order_index,
    }
