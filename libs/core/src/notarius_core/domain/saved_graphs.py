from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from types import MappingProxyType
from typing import Annotated, ClassVar, Literal, Self, cast
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_serializer,
    field_validator,
    model_validator,
)

from notarius_core.domain.errors import SavedGraphRevisionConflictError


GraphIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class SavedGraphValue(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class GraphPoint(SavedGraphValue):
    x: float
    y: float


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("Saved graph config numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise ValueError("Saved graph config object keys must be strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        return tuple(_freeze_json(item) for item in sequence)
    raise ValueError(
        f"Saved graph config values must be JSON-compatible, got {type(value).__name__}"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return {key: _thaw_json(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        sequence = cast(tuple[object, ...], value)
        return [_thaw_json(item) for item in sequence]
    return value


class SavedGraphNode(SavedGraphValue):
    id: GraphIdentifier
    operator_id: GraphIdentifier
    operator_version: int = Field(ge=1)
    config: Mapping[str, object] = Field(default_factory=dict)
    position: GraphPoint

    @field_validator("config")
    @classmethod
    def freeze_config(
        cls,
        value: Mapping[str, object],
    ) -> Mapping[str, object]:
        frozen = _freeze_json(value)
        if not isinstance(frozen, Mapping):
            raise ValueError("Saved graph node config must be an object")
        return cast(Mapping[str, object], frozen)

    @field_serializer("config")
    def serialize_config(self, value: Mapping[str, object]) -> dict[str, object]:
        thawed = _thaw_json(value)
        if not isinstance(thawed, dict):
            raise ValueError("Saved graph node config must be an object")
        return cast(dict[str, object], thawed)

    def config_dict(self) -> dict[str, object]:
        thawed = _thaw_json(self.config)
        if not isinstance(thawed, dict):
            raise ValueError("Saved graph node config must be an object")
        return cast(dict[str, object], thawed)


class SavedGraphProjection(SavedGraphValue):
    path: tuple[GraphIdentifier, ...] = Field(min_length=1)


class SavedGraphConversion(SavedGraphValue):
    id: GraphIdentifier
    version: int = Field(ge=1)


class SavedGraphEdge(SavedGraphValue):
    id: GraphIdentifier
    from_node: GraphIdentifier
    from_port: GraphIdentifier
    to_node: GraphIdentifier
    to_port: GraphIdentifier
    collection_mode: Literal["direct", "map"] = "direct"
    projection: SavedGraphProjection | None = None
    conversion: SavedGraphConversion | None = None
    route_offset: GraphPoint | None = None


class SavedGraphDocument(SavedGraphValue):
    schema_version: Literal[1] = 1
    nodes: tuple[SavedGraphNode, ...] = ()
    edges: tuple[SavedGraphEdge, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Saved graph node ids must be unique")

        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Saved graph edge ids must be unique")

        known_nodes = set(node_ids)
        for edge in self.edges:
            if edge.from_node not in known_nodes:
                raise ValueError(
                    f"Saved graph edge {edge.id} references missing source node "
                    f"{edge.from_node}"
                )
            if edge.to_node not in known_nodes:
                raise ValueError(
                    f"Saved graph edge {edge.id} references missing target node "
                    f"{edge.to_node}"
                )
        return self


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class SavedGraph:
    name: str
    document: SavedGraphDocument
    id: UUID = field(default_factory=uuid4)
    revision: int = 1
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.name = self._validated_name(self.name)
        if self.revision < 1:
            raise ValueError("Saved graph revision must be at least 1")
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("Saved graph timestamps must be timezone-aware")

    def replace(
        self,
        *,
        name: str,
        document: SavedGraphDocument,
        expected_revision: int,
        updated_at: datetime | None = None,
    ) -> None:
        self.ensure_revision(expected_revision)
        validated_name = self._validated_name(name)
        replacement_time = updated_at or _utc_now()
        if replacement_time.tzinfo is None:
            raise ValueError("Saved graph timestamps must be timezone-aware")

        self.name = validated_name
        self.document = document
        self.revision += 1
        self.updated_at = replacement_time

    def ensure_revision(self, expected_revision: int) -> None:
        if expected_revision != self.revision:
            raise SavedGraphRevisionConflictError(
                graph_id=self.id,
                expected_revision=expected_revision,
                actual_revision=self.revision,
            )

    @staticmethod
    def _validated_name(value: str) -> str:
        name = value.strip()
        if name == "":
            raise ValueError("Saved graph name must not be blank")
        if len(name) > 160:
            raise ValueError("Saved graph name must be at most 160 characters")
        return name
