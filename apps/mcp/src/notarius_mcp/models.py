from datetime import datetime
from typing import Annotated, ClassVar, Literal, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)


Identifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]
PortDirection = Literal["input", "output"]
PortShape = Literal["one", "many"]
PluginOrigin = Literal["builtin", "external", "module"]

_MAX_ARTIFACT_CONVERSION_HOPS = 8
_LAYOUT_DIMENSION_MAX = 16_384


class RequestModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class ResponseModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="ignore",
        from_attributes=True,
        allow_inf_nan=False,
    )


class ResultModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


class ArtifactTypeKeyRequest(RequestModel):
    id: str
    schema_version: int = Field(ge=1, strict=True)


class ArtifactTypeKeyResponse(ResponseModel):
    id: str
    schema_version: int = Field(ge=1, strict=True)


class ArtifactTypeBindingRequest(RequestModel):
    variable: Identifier
    artifact_type: ArtifactTypeKeyRequest


class ArtifactTypeBindingResponse(ResponseModel):
    variable: Identifier
    artifact_type: ArtifactTypeKeyResponse


class FieldProjectionResponse(ResponseModel):
    path: list[str]
    target_artifact_type: ArtifactTypeKeyResponse
    title: str


class ArtifactTypeSpecResponse(ResponseModel):
    key: ArtifactTypeKeyResponse
    title: str
    payload_schema: dict[str, JsonValue]
    field_projections: list[FieldProjectionResponse]


class ArtifactConversionKeyResponse(ResponseModel):
    id: str
    version: int = Field(ge=1, strict=True)


class ArtifactConversionSpecResponse(ResponseModel):
    key: ArtifactConversionKeyResponse
    source_artifact_type: ArtifactTypeKeyResponse
    target_artifact_type: ArtifactTypeKeyResponse
    title: str


class PluginSpecResponse(ResponseModel):
    slug: str
    title: str
    origin: PluginOrigin


class PortResponse(ResponseModel):
    name: str
    title: str | None = None
    description: str | None = None
    direction: PortDirection
    artifact_type: ArtifactTypeKeyResponse | None = None
    artifact_type_variable: Identifier | None = None
    shape: PortShape
    accepted_shapes: list[PortShape]
    instance_plugs: bool = False
    variadic: bool = False
    required: bool = True

    @model_validator(mode="after")
    def validate_artifact_type_contract(self) -> Self:
        if (self.artifact_type is None) == (self.artifact_type_variable is None):
            raise ValueError(
                "Port must declare exactly one of artifact_type or "
                "artifact_type_variable"
            )
        return self


class NodeSecretInputResponse(ResponseModel):
    name: str
    config_dependencies: list[str]
    title: str
    description: str | None = None


class NodeSpecResponse(ResponseModel):
    operator_id: str
    operator_version: int = Field(ge=1, strict=True)
    plugin_slug: str
    title: str
    description: str
    config_schema: dict[str, JsonValue]
    input_schema: dict[str, JsonValue]
    output_schema: dict[str, JsonValue]
    inputs: list[PortResponse]
    outputs: list[PortResponse]
    secret_inputs: list[NodeSecretInputResponse] = Field(default_factory=list)
    module_graph_id: UUID | None = None
    module_graph_revision: int | None = Field(default=None, ge=1)
    catalog_visible: bool = True

    @model_validator(mode="after")
    def validate_module_identity(self) -> Self:
        if (self.module_graph_id is None) != (self.module_graph_revision is None):
            raise ValueError(
                "module_graph_id and module_graph_revision must be provided together"
            )
        return self


class UnavailableGraphModuleResponse(ResponseModel):
    graph_id: UUID
    revision: int = Field(ge=1, strict=True)
    name: str
    reason: str


class NodeRegistryResponse(ResponseModel):
    plugins: list[PluginSpecResponse]
    artifact_types: list[ArtifactTypeSpecResponse]
    artifact_conversions: list[ArtifactConversionSpecResponse]
    nodes: list[NodeSpecResponse]
    unavailable_modules: list[UnavailableGraphModuleResponse] = Field(
        default_factory=list
    )


class NodeSearchSummary(ResultModel):
    operator_id: str
    operator_version: int
    plugin_slug: str
    title: str
    description: str
    inputs: list[PortResponse]
    outputs: list[PortResponse]


class NodeSearchResult(ResultModel):
    nodes: list[NodeSearchSummary]
    total_matches: int = Field(ge=0)
    truncated: bool


class NodeInspection(ResultModel):
    node: NodeSpecResponse
    artifact_types: list[ArtifactTypeSpecResponse]
    artifact_conversions: list[ArtifactConversionSpecResponse]


class GraphPointRequest(RequestModel):
    x: float
    y: float


class SavedGraphInputPlugRequest(RequestModel):
    id: Identifier
    port: Identifier


class SavedGraphNodeLayoutRequest(RequestModel):
    width: float | None = Field(default=None, ge=260, le=_LAYOUT_DIMENSION_MAX)
    body_height: float | None = Field(
        default=None,
        ge=96,
        le=_LAYOUT_DIMENSION_MAX,
    )
    appendix_height: float | None = Field(
        default=None,
        ge=120,
        le=_LAYOUT_DIMENSION_MAX,
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


class SavedGraphNodeRequest(RequestModel):
    id: Identifier
    operator_id: Identifier
    operator_version: int = Field(ge=1)
    config: dict[str, JsonValue] = Field(default_factory=dict)
    position: GraphPointRequest
    layout: SavedGraphNodeLayoutRequest | None = None
    input_plugs: list[SavedGraphInputPlugRequest] = Field(default_factory=list)
    artifact_type_bindings: list[ArtifactTypeBindingRequest] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_artifact_type_bindings(self) -> Self:
        variables = [binding.variable for binding in self.artifact_type_bindings]
        if len(variables) != len(set(variables)):
            raise ValueError("Node artifact type binding variables must be unique")
        return self


class SavedGraphProjectionRequest(RequestModel):
    path: list[Identifier] = Field(min_length=1)


class SavedGraphConversionRequest(RequestModel):
    id: Identifier
    version: int = Field(ge=1)


class SavedGraphEdgeRequest(RequestModel):
    id: Identifier
    enabled: bool = True
    from_node: Identifier
    from_port: Identifier
    to_node: Identifier
    to_port: Identifier
    to_plug: Identifier | None = None
    collection_mode: Literal["direct", "map"] = "direct"
    projection: SavedGraphProjectionRequest | None = None
    conversion_path: list[SavedGraphConversionRequest] = Field(
        default_factory=list,
        max_length=_MAX_ARTIFACT_CONVERSION_HOPS,
    )
    route_offset: GraphPointRequest | None = None


class SavedGraphWriteRequest(RequestModel):
    name: str = Field(min_length=1, max_length=160)
    nodes: list[SavedGraphNodeRequest] = Field(default_factory=list)
    edges: list[SavedGraphEdgeRequest] = Field(default_factory=list)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @model_validator(mode="after")
    def validate_structure(self) -> Self:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Saved graph node ids must be unique")

        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("Saved graph edge ids must be unique")

        known_nodes = set(node_ids)
        plugs_by_node: dict[str, dict[str, SavedGraphInputPlugRequest]] = {}
        for node in self.nodes:
            plug_ids = [plug.id for plug in node.input_plugs]
            if len(plug_ids) != len(set(plug_ids)):
                raise ValueError(
                    "Saved graph input plug ids must be unique within node "
                    f"{node.id}"
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


class CreateSavedGraphRequest(SavedGraphWriteRequest):
    pass


class UpdateSavedGraphRequest(SavedGraphWriteRequest):
    expected_revision: int = Field(ge=1)


class GraphPointResponse(ResponseModel):
    x: float
    y: float


class SavedGraphInputPlugResponse(ResponseModel):
    id: Identifier
    port: Identifier


class SavedGraphNodeLayoutResponse(ResponseModel):
    width: float | None = Field(default=None, ge=260, le=_LAYOUT_DIMENSION_MAX)
    body_height: float | None = Field(
        default=None,
        ge=96,
        le=_LAYOUT_DIMENSION_MAX,
    )
    appendix_height: float | None = Field(
        default=None,
        ge=120,
        le=_LAYOUT_DIMENSION_MAX,
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


class SavedGraphNodeResponse(ResponseModel):
    id: Identifier
    operator_id: Identifier
    operator_version: int = Field(ge=1)
    config: dict[str, JsonValue] = Field(default_factory=dict)
    position: GraphPointResponse
    layout: SavedGraphNodeLayoutResponse | None = None
    input_plugs: list[SavedGraphInputPlugResponse] = Field(default_factory=list)
    artifact_type_bindings: list[ArtifactTypeBindingResponse] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_artifact_type_bindings(self) -> Self:
        variables = [binding.variable for binding in self.artifact_type_bindings]
        if len(variables) != len(set(variables)):
            raise ValueError("Node artifact type binding variables must be unique")
        return self


class SavedGraphProjectionResponse(ResponseModel):
    path: list[Identifier] = Field(min_length=1)


class SavedGraphConversionResponse(ResponseModel):
    id: Identifier
    version: int = Field(ge=1)


class SavedGraphEdgeResponse(ResponseModel):
    id: Identifier
    enabled: bool = True
    from_node: Identifier
    from_port: Identifier
    to_node: Identifier
    to_port: Identifier
    to_plug: Identifier | None = None
    collection_mode: Literal["direct", "map"] = "direct"
    projection: SavedGraphProjectionResponse | None = None
    conversion_path: list[SavedGraphConversionResponse] = Field(
        default_factory=list,
        max_length=_MAX_ARTIFACT_CONVERSION_HOPS,
    )
    route_offset: GraphPointResponse | None = None


class SavedGraphResponse(ResponseModel):
    id: UUID
    name: str = Field(min_length=1, max_length=160)
    revision: int = Field(ge=1, strict=True)
    created_at: datetime
    updated_at: datetime
    nodes: list[SavedGraphNodeResponse] = Field(default_factory=list)
    edges: list[SavedGraphEdgeResponse] = Field(default_factory=list)


class SavedGraphSummaryResponse(ResponseModel):
    id: UUID
    name: str
    revision: int = Field(ge=1, strict=True)
    node_count: int = Field(ge=0, strict=True)
    edge_count: int = Field(ge=0, strict=True)
    updated_at: datetime


class SavedGraphListResponse(ResponseModel):
    graphs: list[SavedGraphSummaryResponse]


__all__ = [
    "ArtifactConversionKeyResponse",
    "ArtifactConversionSpecResponse",
    "ArtifactTypeBindingRequest",
    "ArtifactTypeBindingResponse",
    "ArtifactTypeKeyRequest",
    "ArtifactTypeKeyResponse",
    "ArtifactTypeSpecResponse",
    "CreateSavedGraphRequest",
    "FieldProjectionResponse",
    "GraphPointRequest",
    "GraphPointResponse",
    "Identifier",
    "NodeInspection",
    "NodeRegistryResponse",
    "NodeSearchResult",
    "NodeSearchSummary",
    "NodeSecretInputResponse",
    "NodeSpecResponse",
    "PluginOrigin",
    "PluginSpecResponse",
    "PortDirection",
    "PortResponse",
    "PortShape",
    "SavedGraphConversionRequest",
    "SavedGraphConversionResponse",
    "SavedGraphEdgeRequest",
    "SavedGraphEdgeResponse",
    "SavedGraphInputPlugRequest",
    "SavedGraphInputPlugResponse",
    "SavedGraphListResponse",
    "SavedGraphNodeLayoutRequest",
    "SavedGraphNodeLayoutResponse",
    "SavedGraphNodeRequest",
    "SavedGraphNodeResponse",
    "SavedGraphProjectionRequest",
    "SavedGraphProjectionResponse",
    "SavedGraphResponse",
    "SavedGraphSummaryResponse",
    "SavedGraphWriteRequest",
    "UnavailableGraphModuleResponse",
    "UpdateSavedGraphRequest",
]
