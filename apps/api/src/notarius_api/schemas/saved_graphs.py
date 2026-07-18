from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, ClassVar, Literal, Self, cast
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from notarius_core.conversions import MAX_ARTIFACT_CONVERSION_HOPS
from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraph,
    SavedGraphArtifactTypeBinding,
    SavedGraphConversion,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphInputPlug,
    SavedGraphNode,
    SavedGraphNodeLayout,
    SavedGraphProjection,
)

from notarius_api.schemas.workbench import (
    ArtifactTypeBindingModel,
    ArtifactTypeKeyResponse,
)


Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class SavedGraphApiModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class GraphPointModel(SavedGraphApiModel):
    x: float
    y: float


class SavedGraphInputPlugModel(SavedGraphApiModel):
    id: Identifier
    port: Identifier


# Keep in sync with notarius_core.domain.saved_graphs._LAYOUT_DIMENSION_MAX.
_LAYOUT_DIMENSION_MAX = 16_384


class SavedGraphNodeLayoutModel(SavedGraphApiModel):
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


class SavedGraphNodeModel(SavedGraphApiModel):
    id: Identifier
    operator_id: Identifier
    operator_version: int = Field(ge=1)
    config: dict[str, object] = Field(default_factory=dict)
    position: GraphPointModel
    layout: SavedGraphNodeLayoutModel | None = None
    input_plugs: list[SavedGraphInputPlugModel] = Field(default_factory=list)
    artifact_type_bindings: list[ArtifactTypeBindingModel] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_artifact_type_bindings(self) -> Self:
        variables = [binding.variable for binding in self.artifact_type_bindings]
        if len(variables) != len(set(variables)):
            raise ValueError("Node artifact type binding variables must be unique")
        return self


class SavedGraphProjectionModel(SavedGraphApiModel):
    path: list[Identifier] = Field(min_length=1)


class SavedGraphConversionModel(SavedGraphApiModel):
    id: Identifier
    version: int = Field(ge=1)


class SavedGraphEdgeModel(SavedGraphApiModel):
    id: Identifier
    enabled: bool = True
    from_node: Identifier
    from_port: Identifier
    to_node: Identifier
    to_port: Identifier
    to_plug: Identifier | None = None
    collection_mode: Literal["direct", "map"] = "direct"
    projection: SavedGraphProjectionModel | None = None
    conversion_path: list[SavedGraphConversionModel] = Field(
        default_factory=list,
        max_length=MAX_ARTIFACT_CONVERSION_HOPS,
    )
    route_offset: GraphPointModel | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_singular_conversion(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        raw = cast(Mapping[object, object], value)
        if "conversion" not in raw:
            return dict(raw)
        if "conversion_path" in raw:
            raise ValueError(
                "Saved graph edge cannot declare both conversion and conversion_path"
            )
        normalized = dict(raw)
        conversion = normalized.pop("conversion")
        normalized["conversion_path"] = [] if conversion is None else [conversion]
        return normalized


class SavedGraphWriteModel(SavedGraphApiModel):
    name: str = Field(min_length=1, max_length=160)
    nodes: list[SavedGraphNodeModel] = Field(default_factory=list)
    edges: list[SavedGraphEdgeModel] = Field(default_factory=list)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        self.to_document()
        return self

    def to_document(self) -> SavedGraphDocument:
        return SavedGraphDocument(
            nodes=tuple(
                SavedGraphNode(
                    id=node.id,
                    operator_id=node.operator_id,
                    operator_version=node.operator_version,
                    config=node.config,
                    position=GraphPoint(
                        x=node.position.x,
                        y=node.position.y,
                    ),
                    layout=(
                        SavedGraphNodeLayout(
                            width=node.layout.width,
                            body_height=node.layout.body_height,
                            appendix_height=node.layout.appendix_height,
                        )
                        if node.layout is not None
                        else None
                    ),
                    input_plugs=tuple(
                        SavedGraphInputPlug(
                            id=plug.id,
                            port=plug.port,
                        )
                        for plug in node.input_plugs
                    ),
                    artifact_type_bindings=tuple(
                        SavedGraphArtifactTypeBinding(
                            variable=binding.variable,
                            artifact_type=binding.artifact_type.to_key(),
                        )
                        for binding in node.artifact_type_bindings
                    ),
                )
                for node in self.nodes
            ),
            edges=tuple(
                SavedGraphEdge(
                    id=edge.id,
                    enabled=edge.enabled,
                    from_node=edge.from_node,
                    from_port=edge.from_port,
                    to_node=edge.to_node,
                    to_port=edge.to_port,
                    to_plug=edge.to_plug,
                    collection_mode=edge.collection_mode,
                    projection=(
                        SavedGraphProjection(path=tuple(edge.projection.path))
                        if edge.projection is not None
                        else None
                    ),
                    conversion_path=tuple(
                        SavedGraphConversion(
                            id=conversion.id,
                            version=conversion.version,
                        )
                        for conversion in edge.conversion_path
                    ),
                    route_offset=(
                        GraphPoint(
                            x=edge.route_offset.x,
                            y=edge.route_offset.y,
                        )
                        if edge.route_offset is not None
                        else None
                    ),
                )
                for edge in self.edges
            ),
        )


class CreateSavedGraphRequest(SavedGraphWriteModel):
    pass


class UpdateSavedGraphRequest(SavedGraphWriteModel):
    expected_revision: int = Field(ge=1)


class SavedGraphResponse(SavedGraphWriteModel):
    id: UUID
    revision: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_graph(cls, graph: SavedGraph) -> "SavedGraphResponse":
        return cls(
            id=graph.id,
            name=graph.name,
            revision=graph.revision,
            created_at=graph.created_at,
            updated_at=graph.updated_at,
            nodes=[
                SavedGraphNodeModel(
                    id=node.id,
                    operator_id=node.operator_id,
                    operator_version=node.operator_version,
                    config=node.config_dict(),
                    position=GraphPointModel(
                        x=node.position.x,
                        y=node.position.y,
                    ),
                    layout=(
                        SavedGraphNodeLayoutModel(
                            width=node.layout.width,
                            body_height=node.layout.body_height,
                            appendix_height=node.layout.appendix_height,
                        )
                        if node.layout is not None
                        else None
                    ),
                    input_plugs=[
                        SavedGraphInputPlugModel(
                            id=plug.id,
                            port=plug.port,
                        )
                        for plug in node.input_plugs
                    ],
                    artifact_type_bindings=[
                        ArtifactTypeBindingModel(
                            variable=binding.variable,
                            artifact_type=ArtifactTypeKeyResponse(
                                id=binding.artifact_type.id,
                                schema_version=(binding.artifact_type.schema_version),
                            ),
                        )
                        for binding in node.artifact_type_bindings
                    ],
                )
                for node in graph.document.nodes
            ],
            edges=[
                SavedGraphEdgeModel(
                    id=edge.id,
                    enabled=edge.enabled,
                    from_node=edge.from_node,
                    from_port=edge.from_port,
                    to_node=edge.to_node,
                    to_port=edge.to_port,
                    to_plug=edge.to_plug,
                    collection_mode=edge.collection_mode,
                    projection=(
                        SavedGraphProjectionModel(path=list(edge.projection.path))
                        if edge.projection is not None
                        else None
                    ),
                    conversion_path=[
                        SavedGraphConversionModel(
                            id=conversion.id,
                            version=conversion.version,
                        )
                        for conversion in edge.conversion_path
                    ],
                    route_offset=(
                        GraphPointModel(
                            x=edge.route_offset.x,
                            y=edge.route_offset.y,
                        )
                        if edge.route_offset is not None
                        else None
                    ),
                )
                for edge in graph.document.edges
            ],
        )


class SavedGraphSummaryResponse(SavedGraphApiModel):
    id: UUID
    name: str
    revision: int
    node_count: int
    edge_count: int
    updated_at: datetime

    @classmethod
    def from_graph(cls, graph: SavedGraph) -> "SavedGraphSummaryResponse":
        return cls(
            id=graph.id,
            name=graph.name,
            revision=graph.revision,
            node_count=len(graph.document.nodes),
            edge_count=len(graph.document.edges),
            updated_at=graph.updated_at,
        )


class SavedGraphListResponse(SavedGraphApiModel):
    graphs: list[SavedGraphSummaryResponse]
