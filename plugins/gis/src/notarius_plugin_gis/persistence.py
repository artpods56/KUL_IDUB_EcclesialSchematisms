from hashlib import sha256
from io import BytesIO
from typing import cast, final, override

from pydantic import BaseModel

from notarius_core.artifacts import (
    ArtifactObject,
    ArtifactRef,
    ArtifactTypeKey,
    JsonObject,
    UnitOfWorkPort,
)
from notarius_core.domain.errors import NotFoundError
from notarius_core.ports.storage import FileMetadata, FileStoragePort, SaveFileCommand
from notarius_core.runtime.persistence import ArtifactOutputWriter, ArtifactWriteContext
from notarius_core.runtime.resolvers import (
    ArtifactContractError,
    ResolutionError,
    Resolver,
)


@final
class SpatialJsonOutputWriter(ArtifactOutputWriter):
    def __init__(
        self,
        *,
        artifact_type: ArtifactTypeKey,
        model: type[BaseModel],
        content_type: str,
        storage: FileStoragePort,
        uow: UnitOfWorkPort,
        bucket: str,
        storage_backend: str,
    ) -> None:
        self.artifact_type = artifact_type
        self._model = model
        self._content_type = content_type
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
        payload = self._model.model_validate(value)
        content = payload.model_dump_json().encode("utf-8")
        content_hash = sha256(content).hexdigest()
        storage_path = (
            f"{self.artifact_type.id}/v{self.artifact_type.schema_version}/"
            f"{content_hash}.json"
        )
        metadata: FileMetadata = {
            "artifact_kind": self.artifact_type.id,
            "sha256": content_hash,
        }
        if context.node_context.node_id is not None:
            metadata["job_id"] = context.node_context.node_id
        try:
            stored_file = await self._storage.save(
                SaveFileCommand(
                    bucket=self._bucket,
                    path=storage_path,
                    stream=BytesIO(content),
                    content_type=self._content_type,
                    metadata=metadata,
                    allow_overwrite=True,
                )
            )
        except Exception as exc:
            node_id = context.node_context.node_id or "<unknown>"
            raise RuntimeError(
                f"Failed to persist {self.artifact_type.id} output for node "
                f"{node_id!r} at {self._bucket}/{storage_path}"
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
        artifact_metadata: JsonObject = {
            "producer_node_id": context.node_context.node_id,
            "content_hash": content_hash,
            "storage_byte_size": stored_file.byte_size,
            "storage_sha256": stored_file.sha256,
        }
        if provenance:
            artifact_metadata["provenance"] = provenance
        artifact_metadata.update(context.metadata)
        artifact = ArtifactObject(
            artifact_type=self.artifact_type.id,
            schema_version=self.artifact_type.schema_version,
            content_type=self._content_type,
            storage_backend=self._storage_backend,
            bucket=stored_file.bucket,
            object_key=stored_file.path,
            byte_size=stored_file.byte_size,
            sha256=stored_file.sha256,
            metadata=artifact_metadata,
        )
        async with self._uow as uow:
            await uow.artifacts.add(artifact)
            await uow.commit()
        return artifact.ref()


@final
class SpatialJsonResolver[T: BaseModel](Resolver[T]):
    def __init__(
        self,
        *,
        source: ArtifactTypeKey,
        target: type[T],
        uow: UnitOfWorkPort,
        storage: FileStoragePort,
    ) -> None:
        self.source = source
        self.target = cast(type[object], target)
        self._model = target
        self._uow = uow
        self._storage = storage

    @override
    async def resolve(self, ref: ArtifactRef) -> T:
        if ref.key() != self.source:
            raise ArtifactContractError(
                f"Spatial JSON resolver expected {self.source.id}@"
                f"{self.source.schema_version}, got {ref.artifact_type}@"
                f"{ref.schema_version} for {ref.artifact_id}"
            )
        async with self._uow as uow:
            artifact = await uow.artifacts.get(ref.artifact_id)
        if artifact is None:
            raise NotFoundError("Artifact", str(ref.artifact_id))
        if artifact.ref() != ref:
            raise ArtifactContractError(
                f"Artifact repository returned a different artifact ref for {ref.artifact_id}"
            )
        if artifact.bucket is None or artifact.object_key is None:
            raise ArtifactContractError(
                f"Artifact {ref.artifact_id} does not have a storage object"
            )
        try:
            stream = await self._storage.load(
                bucket=artifact.bucket,
                path=artifact.object_key,
            )
            try:
                content = stream.read()
            finally:
                stream.close()
            return self._model.model_validate_json(content)
        except Exception as exc:
            raise ResolutionError(
                f"Failed to resolve spatial artifact {ref.artifact_id} from "
                f"{artifact.bucket}/{artifact.object_key} as {self._model.__name__}"
            ) from exc
