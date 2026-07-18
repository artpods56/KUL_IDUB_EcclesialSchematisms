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

from notarius_core.artifacts import ArtifactTypeKey
from notarius_core.conversions import MAX_ARTIFACT_CONVERSION_HOPS
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


# Shared ceiling for layout axes: common browser/GPU max texture dimension.
# Larger DOM layers can fail to composite. Floors keep node chrome usable.
_LAYOUT_DIMENSION_MAX = 16_384


class SavedGraphNodeLayout(SavedGraphValue):
    """Canvas chrome sizes for a node shell and its artifact appendix."""

    width: float | None = Field(default=None, ge=260, le=_LAYOUT_DIMENSION_MAX)
    body_height: float | None = Field(
        default=None, ge=96, le=_LAYOUT_DIMENSION_MAX
    )
    appendix_height: float | None = Field(
        default=None, ge=120, le=_LAYOUT_DIMENSION_MAX
    )

    @model_validator(mode="after")
    def require_at_least_one_dimension(self) -> Self:
        if (
            self.width is None
            and self.body_height is None
            and self.appendix_height is None
        ):
            raise ValueError(
                "Saved graph node layout must set at least one of width, "
                "body_height, or appendix_height"
            )
        return self


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


class SavedGraphInputPlug(SavedGraphValue):
    id: GraphIdentifier
    port: GraphIdentifier


class SavedGraphArtifactTypeBinding(SavedGraphValue):
    variable: GraphIdentifier
    artifact_type: ArtifactTypeKey

    @field_validator("artifact_type", mode="before")
    @classmethod
    def validate_artifact_type_shape(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        raw = cast(Mapping[object, object], value)
        expected_fields = {"id", "schema_version"}
        if set(raw) != expected_fields:
            raise ValueError(
                "Saved graph artifact type binding must contain exactly id and "
                "schema_version"
            )
        return dict(raw)

    @field_validator("artifact_type")
    @classmethod
    def validate_artifact_type_key(
        cls,
        value: ArtifactTypeKey,
    ) -> ArtifactTypeKey:
        if value.id.strip() == "":
            raise ValueError("Saved graph bound artifact type id must not be empty")
        if value.id != value.id.strip():
            raise ValueError(
                "Saved graph bound artifact type id must not have surrounding "
                "whitespace"
            )
        if len(value.id) > 255:
            raise ValueError(
                "Saved graph bound artifact type id must be at most 255 characters"
            )
        if value.schema_version < 1:
            raise ValueError(
                "Saved graph bound artifact type schema version must be positive"
            )
        return value

    @field_serializer("artifact_type")
    def serialize_artifact_type(
        self,
        value: ArtifactTypeKey,
    ) -> dict[str, object]:
        return {
            "id": value.id,
            "schema_version": value.schema_version,
        }


class SavedGraphNode(SavedGraphValue):
    id: GraphIdentifier
    operator_id: GraphIdentifier
    operator_version: int = Field(ge=1)
    config: Mapping[str, object] = Field(default_factory=dict)
    position: GraphPoint
    layout: SavedGraphNodeLayout | None = None
    input_plugs: tuple[SavedGraphInputPlug, ...] = ()
    artifact_type_bindings: tuple[SavedGraphArtifactTypeBinding, ...] = ()

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

    @model_validator(mode="after")
    def validate_artifact_type_bindings(self) -> Self:
        variables = [binding.variable for binding in self.artifact_type_bindings]
        if len(variables) != len(set(variables)):
            raise ValueError(
                "Saved graph node artifact type binding variables must be unique"
            )
        return self

    def artifact_type_binding_map(self) -> dict[str, ArtifactTypeKey]:
        return {
            binding.variable: binding.artifact_type
            for binding in self.artifact_type_bindings
        }


class SavedGraphProjection(SavedGraphValue):
    path: tuple[GraphIdentifier, ...] = Field(min_length=1)


class SavedGraphConversion(SavedGraphValue):
    id: GraphIdentifier
    version: int = Field(ge=1)


class SavedGraphEdge(SavedGraphValue):
    id: GraphIdentifier
    enabled: bool = True
    from_node: GraphIdentifier
    from_port: GraphIdentifier
    to_node: GraphIdentifier
    to_port: GraphIdentifier
    to_plug: GraphIdentifier | None = None
    collection_mode: Literal["direct", "map"] = "direct"
    projection: SavedGraphProjection | None = None
    conversion_path: tuple[SavedGraphConversion, ...] = Field(
        default=(),
        max_length=MAX_ARTIFACT_CONVERSION_HOPS,
    )
    route_offset: GraphPoint | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_singular_conversion(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        raw = cast(Mapping[object, object], value)
        if "conversion" not in raw:
            return dict(raw)
        if "conversion_path" in raw:
            raise ValueError(
                "Saved graph edge cannot declare both conversion and conversion_path"
            )
        migrated = dict(raw)
        conversion = migrated.pop("conversion")
        migrated["conversion_path"] = [] if conversion is None else [conversion]
        return migrated


class SavedGraphDocument(SavedGraphValue):
    schema_version: Literal[3] = 3
    nodes: tuple[SavedGraphNode, ...] = ()
    edges: tuple[SavedGraphEdge, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def migrate_document(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        raw = cast(Mapping[object, object], value)
        if raw.get("schema_version", 1) not in (1, 2):
            return dict(raw)
        migrated = dict(raw)
        migrated["schema_version"] = 3
        return migrated

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Saved graph node ids must be unique")

        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Saved graph edge ids must be unique")

        known_nodes = set(node_ids)
        plugs_by_node: dict[str, dict[str, SavedGraphInputPlug]] = {}
        for node in self.nodes:
            plug_ids = [plug.id for plug in node.input_plugs]
            if len(plug_ids) != len(set(plug_ids)):
                raise ValueError(
                    f"Saved graph input plug ids must be unique within node {node.id}"
                )
            plugs_by_node[node.id] = {plug.id: plug for plug in node.input_plugs}

        connected_plugs: set[tuple[str, str]] = set()
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
            if edge.to_plug is None:
                continue
            target_plug = plugs_by_node[edge.to_node].get(edge.to_plug)
            if target_plug is None:
                raise ValueError(
                    f"Saved graph edge {edge.id} references missing input plug "
                    f"{edge.to_plug} on target node {edge.to_node}"
                )
            if target_plug.port != edge.to_port:
                raise ValueError(
                    f"Saved graph edge {edge.id} targets port {edge.to_port}, but "
                    f"input plug {edge.to_plug} belongs to port {target_plug.port}"
                )
            plug_key = (edge.to_node, edge.to_plug)
            if plug_key in connected_plugs:
                raise ValueError(
                    f"Saved graph input plug {edge.to_plug} on node "
                    f"{edge.to_node} accepts at most one edge"
                )
            connected_plugs.add(plug_key)
        return self


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validated_graph_name(value: str) -> str:
    name = value.strip()
    if name == "":
        raise ValueError("Saved graph name must not be blank")
    if len(name) > 160:
        raise ValueError("Saved graph name must be at most 160 characters")
    return name


@dataclass
class SavedGraph:
    name: str
    document: SavedGraphDocument
    id: UUID = field(default_factory=uuid4)
    revision: int = 1
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.name = _validated_graph_name(self.name)
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
        validated_name = _validated_graph_name(name)
        replacement_time = updated_at or _utc_now()
        if replacement_time.tzinfo is None:
            raise ValueError("Saved graph timestamps must be timezone-aware")

        self.name = validated_name
        self.document = document
        self.revision += 1
        self.updated_at = replacement_time

    def snapshot(self) -> "SavedGraphRevision":
        return SavedGraphRevision(
            graph_id=self.id,
            revision=self.revision,
            name=self.name,
            document=self.document,
            created_at=self.updated_at,
        )

    def ensure_revision(self, expected_revision: int) -> None:
        if expected_revision != self.revision:
            raise SavedGraphRevisionConflictError(
                graph_id=self.id,
                expected_revision=expected_revision,
                actual_revision=self.revision,
            )


@dataclass(frozen=True, slots=True)
class SavedGraphRevision:
    graph_id: UUID
    revision: int
    name: str
    document: SavedGraphDocument
    created_at: datetime

    @property
    def id(self) -> UUID:
        return self.graph_id

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validated_graph_name(self.name))
        if self.revision < 1:
            raise ValueError("Saved graph snapshot revision must be at least 1")
        if self.created_at.tzinfo is None:
            raise ValueError("Saved graph snapshot timestamp must be timezone-aware")
