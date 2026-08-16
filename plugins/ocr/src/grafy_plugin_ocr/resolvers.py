from io import BytesIO
from typing import final, override
from uuid import UUID

from PIL import Image as ImageModule
from PIL.Image import Image

from grafy_core.artifacts import ArtifactRef, UnitOfWorkPort
from grafy_core.domain.errors import NotFoundError
from grafy_core.operators.images import RASTER_IMAGE
from grafy_core.ports.storage import FileStoragePort
from grafy_core.runtime.resolvers import (
    ArtifactContractError,
    ResolutionError,
    Resolver,
)

from grafy_plugin_ocr.mistral import EncodedPageImage


@final
class EncodedPageImageResolver(Resolver[EncodedPageImage]):
    source = RASTER_IMAGE.key
    target = EncodedPageImage

    def __init__(self, uow: UnitOfWorkPort, storage: FileStoragePort) -> None:
        self._uow = uow
        self._storage = storage

    @override
    async def resolve(
        self,
        ref: ArtifactRef,
        workspace_id: UUID,
    ) -> EncodedPageImage:
        if ref.key() != self.source:
            message = (
                f"Encoded page resolver expected {self.source.id}@"
                f"{self.source.schema_version}, got {ref.artifact_type}@"
                f"{ref.schema_version} for {ref.artifact_id}"
            )
            raise ArtifactContractError(message)

        async with self._uow as uow:
            artifact = await uow.artifacts.get(workspace_id, ref.artifact_id)
        if artifact is None:
            raise NotFoundError("Artifact", str(ref.artifact_id))
        if artifact.ref() != ref:
            message = (
                "Artifact repository returned a different artifact ref for "
                f"{ref.artifact_id}"
            )
            raise ArtifactContractError(message)
        if artifact.bucket is None or artifact.object_key is None:
            message = f"Artifact {ref.artifact_id} does not have a storage object"
            raise ArtifactContractError(message)

        try:
            stream = await self._storage.load(
                bucket=artifact.bucket,
                path=artifact.object_key,
            )
            try:
                content = stream.read()
            finally:
                stream.close()
        except Exception as exc:
            message = (
                f"Failed to load encoded source image {ref.artifact_id} from "
                f"{artifact.bucket}/{artifact.object_key}"
            )
            raise ResolutionError(message) from exc

        original_filename = artifact.metadata.get("original_filename")
        filename = (
            original_filename
            if isinstance(original_filename, str) and original_filename != ""
            else str(ref.artifact_id)
        )
        return EncodedPageImage(
            artifact_id=ref.artifact_id,
            filename=filename,
            content=content,
            content_type=artifact.content_type,
        )


@final
class PilImageResolver(Resolver[Image]):
    source = RASTER_IMAGE.key
    target = Image

    def __init__(self, uow: UnitOfWorkPort, storage: FileStoragePort) -> None:
        self._uow = uow
        self._storage = storage

    @override
    async def resolve(
        self,
        ref: ArtifactRef,
        workspace_id: UUID,
    ) -> Image:
        if ref.key() != self.source:
            message = (
                f"PIL image resolver expected {self.source.id}@"
                f"{self.source.schema_version}, got {ref.artifact_type}@"
                f"{ref.schema_version} for {ref.artifact_id}"
            )
            raise ArtifactContractError(message)

        async with self._uow as uow:
            artifact = await uow.artifacts.get(workspace_id, ref.artifact_id)

        if artifact is None:
            raise NotFoundError("Artifact", str(ref.artifact_id))

        if artifact.ref() != ref:
            message = (
                "Artifact repository returned a different artifact ref for "
                f"{ref.artifact_id}"
            )
            raise ArtifactContractError(message)

        if artifact.bucket is None or artifact.object_key is None:
            message = f"Artifact {ref.artifact_id} does not have a storage object"
            raise ArtifactContractError(message)

        try:
            stream = await self._storage.load(
                bucket=artifact.bucket,
                path=artifact.object_key,
            )
            try:
                image_bytes = stream.read()
            finally:
                stream.close()

            with ImageModule.open(BytesIO(image_bytes)) as image:
                return image.copy()
        except Exception as exc:
            message = (
                f"Failed to resolve artifact {ref.artifact_id} from storage object "
                f"{artifact.bucket}/{artifact.object_key} as {self.target.__name__}"
            )
            raise ResolutionError(message) from exc
