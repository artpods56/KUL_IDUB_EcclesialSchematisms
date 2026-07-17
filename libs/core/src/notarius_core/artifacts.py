from asyncio import Lock
from collections.abc import Collection
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Literal, Protocol, Self, TypeAlias, final, override
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from notarius_core.domain.invocation_cache import InvocationCacheEntry
    from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
    from notarius_core.ports.invocation_cache import InvocationCacheRepositoryPort
    from notarius_core.ports.materialized_outputs import (
        MaterializedNodeOutputsRepositoryPort,
    )

JsonObject: TypeAlias = dict[str, object]
MaterializedJsonType: TypeAlias = Literal["string", "integer"]


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
    materialized_json_type: MaterializedJsonType | None = None


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


@dataclass
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

    async def get_many(
        self,
        artifact_ids: Collection[UUID],
    ) -> dict[UUID, ArtifactObject]: ...

    async def remove(self, artifact: ArtifactObject) -> None: ...

    async def list_by_type(self, key: ArtifactTypeKey) -> list[ArtifactObject]: ...


@dataclass(slots=True)
class InMemoryDataStore:
    artifacts: dict[UUID, ArtifactObject] = field(default_factory=dict)
    materialized_outputs: dict[
        tuple[UUID, int, str],
        "MaterializedNodeOutputs",
    ] = field(default_factory=dict)
    invocation_cache: dict[str, "InvocationCacheEntry"] = field(default_factory=dict)

    def clone(self) -> Self:
        return _clone(self)

    def replace_with(self, other: Self) -> None:
        self.artifacts = _clone(other.artifacts)
        self.materialized_outputs = _clone(other.materialized_outputs)
        self.invocation_cache = _clone(other.invocation_cache)


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
    async def get_many(
        self,
        artifact_ids: Collection[UUID],
    ) -> dict[UUID, ArtifactObject]:
        return {
            artifact_id: artifact
            for artifact_id in artifact_ids
            if (artifact := self._store.artifacts.get(artifact_id)) is not None
        }

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


@final
class InMemoryMaterializedNodeOutputsRepository:
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    async def upsert(self, value: "MaterializedNodeOutputs") -> None:
        key = (value.graph_id, value.graph_revision, value.node_id)
        self._store.materialized_outputs[key] = _clone(value)

    async def get(
        self,
        graph_id: UUID,
        graph_revision: int,
        node_id: str,
    ) -> "MaterializedNodeOutputs | None":
        value = self._store.materialized_outputs.get(
            (graph_id, graph_revision, node_id)
        )
        return _clone(value) if value is not None else None

    async def list_for_graph(
        self,
        graph_id: UUID,
        graph_revision: int,
    ) -> list["MaterializedNodeOutputs"]:
        values = [
            value
            for (saved_graph_id, saved_revision, _), value in (
                self._store.materialized_outputs.items()
            )
            if saved_graph_id == graph_id and saved_revision == graph_revision
        ]
        return _clone(sorted(values, key=lambda value: value.node_id))


@final
class InMemoryInvocationCacheRepository:
    def __init__(self, store: InMemoryDataStore) -> None:
        self._store = store

    async def get(self, key_sha256: str) -> "InvocationCacheEntry | None":
        entry = self._store.invocation_cache.get(key_sha256)
        return _clone(entry) if entry is not None else None

    async def put_if_absent(self, entry: "InvocationCacheEntry") -> bool:
        if entry.key_sha256 in self._store.invocation_cache:
            return False
        self._store.invocation_cache[entry.key_sha256] = _clone(entry)
        return True

    async def remove_if_current(
        self,
        key_sha256: str,
        generation: UUID,
    ) -> bool:
        entry = self._store.invocation_cache.get(key_sha256)
        if entry is None or entry.generation != generation:
            return False
        del self._store.invocation_cache[key_sha256]
        return True


@dataclass(frozen=True, slots=True)
class _InMemoryUnitOfWorkState:
    working_store: InMemoryDataStore
    artifacts: ArtifactRepositoryPort
    materialized_outputs: "MaterializedNodeOutputsRepositoryPort"
    invocation_cache: "InvocationCacheRepositoryPort"


class InMemoryUnitOfWork(UnitOfWorkPort):
    def __init__(self, store: InMemoryDataStore | None = None) -> None:
        self._store = store or InMemoryDataStore()
        self._transaction_lock = Lock()
        self._state: ContextVar[_InMemoryUnitOfWorkState | None] = ContextVar(
            "notarius_in_memory_unit_of_work_state",
            default=None,
        )
        self.commit_count = 0
        self.rollback_count = 0

    @property
    @override
    def artifacts(self) -> ArtifactRepositoryPort:
        return self._entered_state().artifacts

    @property
    def materialized_outputs(self) -> "MaterializedNodeOutputsRepositoryPort":
        return self._entered_state().materialized_outputs

    @property
    def invocation_cache(self) -> "InvocationCacheRepositoryPort":
        return self._entered_state().invocation_cache

    @override
    async def __aenter__(self) -> Self:
        if self._state.get() is not None:
            raise RuntimeError("Unit of work is already entered in this task")

        await self._transaction_lock.acquire()
        try:
            working_store = self._store.clone()
            self._state.set(
                _InMemoryUnitOfWorkState(
                    working_store=working_store,
                    artifacts=InMemoryArtifactRepository(working_store),
                    materialized_outputs=InMemoryMaterializedNodeOutputsRepository(
                        working_store
                    ),
                    invocation_cache=InMemoryInvocationCacheRepository(working_store),
                )
            )
        except BaseException:
            self._transaction_lock.release()
            raise
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        state = self._state.get()
        if state is None:
            return
        try:
            if exc_type is not None:
                await self.rollback()
        finally:
            self._state.set(None)
            self._transaction_lock.release()

    @override
    async def commit(self) -> None:
        state = self._entered_state()
        self._store.replace_with(state.working_store)
        self.commit_count += 1

    @override
    async def rollback(self) -> None:
        state = self._entered_state()
        state.working_store.replace_with(self._store)
        self.rollback_count += 1

    def _entered_state(self) -> _InMemoryUnitOfWorkState:
        state = self._state.get()
        if state is None:
            raise RuntimeError("Unit of work is not entered")
        return state
