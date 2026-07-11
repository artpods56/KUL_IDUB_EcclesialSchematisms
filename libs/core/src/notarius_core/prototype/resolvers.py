from io import BytesIO
from typing import Protocol, cast, final, override

from PIL import Image as ImageModule
from PIL.Image import Image
from pydantic import BaseModel

from notarius_core.domain.errors import NotFoundError
from notarius_core.ports.storage import FileStoragePort
from notarius_core.prototype.artifacts import (
    SOURCE_PAGE_IMAGE,
    ArtifactRef,
    ArtifactTypeKey,
    UnitOfWorkPort,
)
from notarius_core.prototype.mistral_ocr import EncodedPageImage


class ResolutionError(RuntimeError):
    pass


class UnknownResolverError(ResolutionError):
    pass


class ArtifactContractError(ResolutionError):
    pass


class Resolver[T](Protocol):
    source: ArtifactTypeKey
    target: type[object]

    async def resolve(
        self,
        ref: ArtifactRef,
    ) -> T: ...


class ResolverRegistry:
    def __init__(self, resolvers: list[Resolver[object]] | None = None) -> None:
        self._resolvers: dict[
            tuple[ArtifactTypeKey, type[object]], Resolver[object]
        ]
        self._resolvers = {}
        for resolver in resolvers or []:
            self.register(resolver)

    def register(self, resolver: Resolver[object]) -> None:
        self._resolvers[(resolver.source, resolver.target)] = resolver

    async def resolve[T](
        self,
        ref: ArtifactRef,
        target: type[T],
    ) -> T:
        resolver = self._resolvers.get((ref.key(), target))
        if resolver is None:
            message = (
                f"No resolver registered for {ref.artifact_type}@"
                f"{ref.schema_version} as {target}"
            )
            raise UnknownResolverError(message)
        return cast(T, await resolver.resolve(ref))


@final
class InlineModelResolver[T: BaseModel](Resolver[T]):
    """Materializes an inline JSON artifact as its declared Pydantic model."""

    def __init__(
        self,
        *,
        source: ArtifactTypeKey,
        target: type[T],
        uow: UnitOfWorkPort,
    ) -> None:
        self.source = source
        self.target = cast(type[object], target)
        self._model = target
        self._uow = uow

    @override
    async def resolve(self, ref: ArtifactRef) -> T:
        if ref.key() != self.source:
            message = (
                f"Inline model resolver expected {self.source.id}@"
                f"{self.source.schema_version}, got {ref.artifact_type}@"
                f"{ref.schema_version} for {ref.artifact_id}"
            )
            raise ArtifactContractError(message)

        async with self._uow as uow:
            artifact = await uow.artifacts.get(ref.artifact_id)
        if artifact is None:
            raise NotFoundError("Artifact", str(ref.artifact_id))
        if artifact.ref() != ref:
            message = f"Artifact repository returned a different artifact ref for {ref.artifact_id}"
            raise ArtifactContractError(message)
        if artifact.inline_payload is None:
            message = f"Artifact {ref.artifact_id} does not have an inline JSON payload"
            raise ArtifactContractError(message)

        try:
            return self._model.model_validate(artifact.inline_payload)
        except Exception as exc:
            message = (
                f"Failed to resolve artifact {ref.artifact_id} as "
                f"{self._model.__module__}.{self._model.__qualname__}"
            )
            raise ResolutionError(message) from exc


@final
class EncodedPageImageResolver(Resolver[EncodedPageImage]):
    source = SOURCE_PAGE_IMAGE.key
    target = EncodedPageImage

    def __init__(self, uow: UnitOfWorkPort, storage: FileStoragePort) -> None:
        self._uow = uow
        self._storage = storage

    @override
    async def resolve(self, ref: ArtifactRef) -> EncodedPageImage:
        if ref.key() != self.source:
            message = (
                f"Encoded page resolver expected {self.source.id}@"
                f"{self.source.schema_version}, got {ref.artifact_type}@"
                f"{ref.schema_version} for {ref.artifact_id}"
            )
            raise ArtifactContractError(message)

        async with self._uow as uow:
            artifact = await uow.artifacts.get(ref.artifact_id)
        if artifact is None:
            raise NotFoundError("Artifact", str(ref.artifact_id))
        if artifact.ref() != ref:
            message = f"Artifact repository returned a different artifact ref for {ref.artifact_id}"
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
    source = SOURCE_PAGE_IMAGE.key
    target = Image

    def __init__(self, uow: UnitOfWorkPort, storage: FileStoragePort) -> None:
        self._uow = uow
        self._storage = storage

    @override
    async def resolve(
        self,
        ref: ArtifactRef,
    ) -> Image:
        if ref.key() != self.source:
            message = (
                f"PIL image resolver expected {self.source.id}@"
                f"{self.source.schema_version}, got {ref.artifact_type}@"
                f"{ref.schema_version} for {ref.artifact_id}"
            )
            raise ArtifactContractError(message)

        async with self._uow as uow:
            artifact = await uow.artifacts.get(ref.artifact_id)

        if artifact is None:
            raise NotFoundError("Artifact", str(ref.artifact_id))

        if artifact.ref() != ref:
            message = f"Artifact repository returned a different artifact ref for {ref.artifact_id}"
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
