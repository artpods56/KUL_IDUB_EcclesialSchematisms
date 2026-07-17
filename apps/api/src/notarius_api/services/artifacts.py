import json
from uuid import UUID

from notarius_core.artifacts import (
    ArtifactObject,
    ArtifactRefSequence,
    UnitOfWorkPort,
)
from notarius_core.domain.artifact_outputs import ArtifactOutputValue
from notarius_core.ports.storage import FileStoragePort

from notarius_api.services.errors import WorkbenchOperationError


class ArtifactService:
    """Loads persisted artifacts and validates graph-facing artifact references."""

    def __init__(
        self,
        unit_of_work: UnitOfWorkPort,
        storage: FileStoragePort,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._storage = storage

    async def get(self, artifact_id: UUID) -> ArtifactObject | None:
        async with self._unit_of_work as unit_of_work:
            return await unit_of_work.artifacts.get(artifact_id)

    async def load_content(self, artifact: ArtifactObject) -> bytes:
        if artifact.inline_payload is not None:
            return (
                json.dumps(
                    artifact.inline_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
        if artifact.bucket is None or artifact.object_key is None:
            raise WorkbenchOperationError(
                f"Artifact {artifact.id} has no stored payload"
            )
        stream = await self._storage.load(
            bucket=artifact.bucket,
            path=artifact.object_key,
        )
        try:
            return stream.read()
        finally:
            stream.close()

    async def is_accessible(self, value: ArtifactOutputValue) -> bool:
        refs = value.item_refs if isinstance(value, ArtifactRefSequence) else (value,)
        for ref in refs:
            artifact = await self.get(ref.artifact_id)
            if artifact is None or artifact.ref() != ref:
                return False
            if artifact.inline_payload is not None:
                continue
            if artifact.bucket is None or artifact.object_key is None:
                return False
            if not self._storage.exists(artifact.bucket, artifact.object_key):
                return False
        return True

    async def validate_refs(
        self,
        value: ArtifactOutputValue,
        *,
        context: str,
    ) -> None:
        refs = value.item_refs if isinstance(value, ArtifactRefSequence) else (value,)
        for index, ref in enumerate(refs):
            item_context = (
                f" sequence item {index}"
                if isinstance(value, ArtifactRefSequence)
                else ""
            )
            artifact = await self.get(ref.artifact_id)
            if artifact is None:
                raise WorkbenchOperationError(
                    f"{context}{item_context} references missing artifact "
                    f"{ref.artifact_id}"
                )
            if artifact.ref() != ref:
                raise WorkbenchOperationError(
                    f"{context}{item_context} does not match the repository ref "
                    f"for artifact {ref.artifact_id}"
                )


__all__ = ["ArtifactService"]
