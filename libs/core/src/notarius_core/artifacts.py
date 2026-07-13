from copy import deepcopy
from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol, Self, TypeAlias, final, override
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonObject: TypeAlias = dict[str, object]


class NodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NodeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class NodeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoConfig(NodeConfig):
    pass


def _clone[T](value: T) -> T:
    return deepcopy(value)


def artifact_id() -> UUID:
    return uuid4()


def sequence_id() -> UUID:
    return uuid4()


@dataclass(frozen=True, slots=True)
class ArtifactTypeKey:
    id: str
    schema_version: int


@dataclass(frozen=True, slots=True)
class ArtifactFieldProjection:
    path: tuple[str, ...]
    target: ArtifactTypeKey
    title: str


@dataclass(frozen=True, slots=True)
class ArtifactTypeSpec:
    key: ArtifactTypeKey
    title: str
    payload_schema: JsonObject = field(default_factory=dict)
    field_projections: tuple[ArtifactFieldProjection, ...] = ()

SOURCE_PAGE_IMAGE = ArtifactTypeSpec(
    key=ArtifactTypeKey("source.page_image", 1),
    title="Source page image",
)

TABLE_FRAGMENT = ArtifactTypeSpec(
    key=ArtifactTypeKey("table.fragment", 1),
    title="Extracted table fragment",
)

TABLE_PAGE = ArtifactTypeSpec(
    key=ArtifactTypeKey("table.page", 1),
    title="Merged page table",
)

TABLE_CSV_BUNDLE = ArtifactTypeSpec(
    key=ArtifactTypeKey("tabular.csv_bundle", 1),
    title="CSV export bundle",
)


class ArtifactRef(BaseModel):
    artifact_id: UUID
    artifact_type: str
    schema_version: int
    content_hash: str | None = None

    @classmethod
    def from_key(
        cls,
        *,
        artifact_id: UUID,
        key: ArtifactTypeKey,
        content_hash: str | None = None,
    ) -> Self:
        return cls(
            artifact_id=artifact_id,
            artifact_type=key.id,
            schema_version=key.schema_version,
            content_hash=content_hash,
        )

    def key(self) -> ArtifactTypeKey:
        return ArtifactTypeKey(self.artifact_type, self.schema_version)


class ArtifactRefSequence(BaseModel):
    sequence_id: UUID = Field(default_factory=sequence_id)
    artifact_type: str
    schema_version: int
    item_refs: list[ArtifactRef]
    ordered: bool = True
    index_key: str = "order_index"
    metadata: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_key(
        cls,
        *,
        key: ArtifactTypeKey,
        item_refs: list[ArtifactRef],
        metadata: JsonObject | None = None,
    ) -> Self:
        return cls(
            artifact_type=key.id,
            schema_version=key.schema_version,
            item_refs=item_refs,
            metadata=metadata or {},
        )

    @model_validator(mode="after")
    def validate_item_refs(self) -> Self:
        for item_ref in self.item_refs:
            if item_ref.artifact_type != self.artifact_type:
                message = (
                    f"ArtifactSequence item type mismatch: expected "
                    f"{self.artifact_type}, got {item_ref.artifact_type}"
                )
                raise ValueError(message)
            if item_ref.schema_version != self.schema_version:
                message = (
                    f"ArtifactSequence schema version mismatch: expected "
                    f"{self.schema_version}, got {item_ref.schema_version}"
                )
                raise ValueError(message)
        return self


@dataclass(slots=True)
class ArtifactObject:
    artifact_type: str
    schema_version: int
    content_type: str
    id: UUID = field(default_factory=artifact_id)
    storage_backend: str = "local"
    bucket: str | None = None
    object_key: str | None = None
    inline_payload: JsonObject | None = None
    byte_size: int | None = None
    sha256: str | None = None
    metadata: JsonObject = field(default_factory=dict)

    def ref(self) -> ArtifactRef:
        return ArtifactRef.from_key(
            artifact_id=self.id,
            key=ArtifactTypeKey(self.artifact_type, self.schema_version),
            content_hash=self.sha256,
        )


class ArtifactRepositoryPort(Protocol):
    async def add(self, artifact: ArtifactObject) -> None: ...

    async def get(self, artifact_id: UUID) -> ArtifactObject | None: ...

    async def remove(self, artifact: ArtifactObject) -> None: ...

    async def list_by_type(self, key: ArtifactTypeKey) -> list[ArtifactObject]: ...


@dataclass(slots=True)
class InMemoryDataStore:
    artifacts: dict[UUID, ArtifactObject] = field(default_factory=dict)

    def clone(self) -> Self:
        return _clone(self)

    def replace_with(self, other: Self) -> None:
        self.artifacts = _clone(other.artifacts)


class UnitOfWorkPort(Protocol):
    @property
    def artifacts(self) -> ArtifactRepositoryPort: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


@final
class InMemoryArtifactRepository(ArtifactRepositoryPort):
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    @override
    async def add(self, artifact: ArtifactObject) -> None:
        self._store.artifacts[artifact.id] = artifact

    @override
    async def get(self, artifact_id: UUID) -> ArtifactObject | None:
        return self._store.artifacts.get(artifact_id)

    @override
    async def remove(self, artifact: ArtifactObject) -> None:
        self._store.artifacts.pop(artifact.id, None)

    @override
    async def list_by_type(self, key: ArtifactTypeKey) -> list[ArtifactObject]:
        return [
            artifact
            for artifact in self._store.artifacts.values()
            if artifact.artifact_type == key.id
            and artifact.schema_version == key.schema_version
        ]


class InMemoryUnitOfWork(UnitOfWorkPort):
    def __init__(self, store: InMemoryDataStore | None = None) -> None:
        self._store = store or InMemoryDataStore()
        self._working_store: InMemoryDataStore | None = None
        self._artifacts: ArtifactRepositoryPort | None = None
        self.commit_count = 0
        self.rollback_count = 0

    @property
    @override
    def artifacts(self) -> ArtifactRepositoryPort:
        if self._artifacts is None:
            raise RuntimeError("Unit of work is not entered")
        return self._artifacts

    @override
    async def __aenter__(self) -> Self:
        if self._working_store is not None:
            raise RuntimeError("Unit of work is already entered")

        working_store = self._store.clone()
        self._working_store = working_store
        self._artifacts = InMemoryArtifactRepository(working_store)
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        if exc_type is not None and self._working_store is not None:
            await self.rollback()
        self._working_store = None
        self._artifacts = None

    @override
    async def commit(self) -> None:
        if self._working_store is None:
            raise RuntimeError("Unit of work is not entered")
        self._store.replace_with(self._working_store)
        self.commit_count += 1

    @override
    async def rollback(self) -> None:
        if self._working_store is None:
            raise RuntimeError("Unit of work is not entered")
        self._working_store.replace_with(self._store)
        self.rollback_count += 1
