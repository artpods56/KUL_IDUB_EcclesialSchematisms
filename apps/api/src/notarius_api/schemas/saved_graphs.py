from datetime import datetime
from typing import Annotated, ClassVar, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from notarius_core.domain.saved_graphs import (
    GraphPoint,
    SavedGraph,
    SavedGraphConversion,
    SavedGraphDocument,
    SavedGraphEdge,
    SavedGraphNode,
    SavedGraphProjection,
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


class SavedGraphNodeModel(SavedGraphApiModel):
    id: Identifier
    operator_id: Identifier
    operator_version: int = Field(ge=1)
    config: dict[str, object] = Field(default_factory=dict)
    position: GraphPointModel


class SavedGraphProjectionModel(SavedGraphApiModel):
    path: list[Identifier] = Field(min_length=1)


class SavedGraphConversionModel(SavedGraphApiModel):
    id: Identifier
    version: int = Field(ge=1)


class SavedGraphEdgeModel(SavedGraphApiModel):
    id: Identifier
    from_node: Identifier
    from_port: Identifier
    to_node: Identifier
    to_port: Identifier
    collection_mode: Literal["direct", "map"] = "direct"
    projection: SavedGraphProjectionModel | None = None
    conversion: SavedGraphConversionModel | None = None
    route_offset: GraphPointModel | None = None


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
                )
                for node in self.nodes
            ),
            edges=tuple(
                SavedGraphEdge(
                    id=edge.id,
                    from_node=edge.from_node,
                    from_port=edge.from_port,
                    to_node=edge.to_node,
                    to_port=edge.to_port,
                    collection_mode=edge.collection_mode,
                    projection=(
                        SavedGraphProjection(path=tuple(edge.projection.path))
                        if edge.projection is not None
                        else None
                    ),
                    conversion=(
                        SavedGraphConversion(
                            id=edge.conversion.id,
                            version=edge.conversion.version,
                        )
                        if edge.conversion is not None
                        else None
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
                )
                for node in graph.document.nodes
            ],
            edges=[
                SavedGraphEdgeModel(
                    id=edge.id,
                    from_node=edge.from_node,
                    from_port=edge.from_port,
                    to_node=edge.to_node,
                    to_port=edge.to_port,
                    collection_mode=edge.collection_mode,
                    projection=(
                        SavedGraphProjectionModel(path=list(edge.projection.path))
                        if edge.projection is not None
                        else None
                    ),
                    conversion=(
                        SavedGraphConversionModel(
                            id=edge.conversion.id,
                            version=edge.conversion.version,
                        )
                        if edge.conversion is not None
                        else None
                    ),
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
