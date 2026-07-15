from typing import Annotated, ClassVar, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence
from notarius_core.nodes import PortShape


PortDirection = Literal["input", "output"]


class ApiResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)


class ArtifactTypeKeyResponse(ApiResponse):
    id: str
    schema_version: int


class FieldProjectionResponse(ApiResponse):
    path: list[str]
    target_artifact_type: ArtifactTypeKeyResponse
    title: str


class ArtifactTypeSpecResponse(ApiResponse):
    key: ArtifactTypeKeyResponse
    title: str
    payload_schema: dict[str, object]
    field_projections: list[FieldProjectionResponse]


class ArtifactConversionKeyResponse(ApiResponse):
    id: str
    version: int


class ArtifactConversionSpecResponse(ApiResponse):
    key: ArtifactConversionKeyResponse
    source_artifact_type: ArtifactTypeKeyResponse
    target_artifact_type: ArtifactTypeKeyResponse
    title: str


class PluginSpecResponse(ApiResponse):
    slug: str
    title: str


class PortResponse(ApiResponse):
    name: str
    title: str | None = None
    description: str | None = None
    direction: PortDirection
    artifact_type: ArtifactTypeKeyResponse
    shape: PortShape
    variadic: bool = False
    required: bool = True


class NodeSpecResponse(ApiResponse):
    operator_id: str
    operator_version: int
    plugin_slug: str
    title: str
    description: str
    config_schema: dict[str, object]
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    inputs: list[PortResponse]
    outputs: list[PortResponse]


class NodeRegistryResponse(ApiResponse):
    plugins: list[PluginSpecResponse]
    artifact_types: list[ArtifactTypeSpecResponse]
    artifact_conversions: list[ArtifactConversionSpecResponse]
    nodes: list[NodeSpecResponse]


class UploadRequest(BaseModel):
    filename: str
    content_base64: str


class SampleRequest(BaseModel):
    count: int = Field(default=2, ge=1, le=8)


class SelectionItemResponse(ApiResponse):
    connector_id: str
    external_uri: str
    display_name: str
    size_bytes: int


class RunNodeRequest(BaseModel):
    id: str
    operator_id: str
    operator_version: int
    config: dict[str, object] = Field(default_factory=dict)


class FieldProjectionRequest(BaseModel):
    path: list[str] = Field(min_length=1)


class ArtifactConversionRequest(BaseModel):
    id: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1),
    ]
    version: int = Field(ge=1)


class RunEdgeRequest(BaseModel):
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    projection: FieldProjectionRequest | None = None
    conversion: ArtifactConversionRequest | None = None
    collection_mode: Literal["direct", "map"] = "direct"


class PinnedOutputRequest(BaseModel):
    from_node: str
    from_port: str
    value: ArtifactRef | ArtifactRefSequence


class RunRequest(BaseModel):
    nodes: list[RunNodeRequest]
    edges: list[RunEdgeRequest] = Field(default_factory=list)
    pinned_outputs: list[PinnedOutputRequest] = Field(default_factory=list)
    graph_id: UUID | None = None
    graph_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_graph_context(self) -> Self:
        if (self.graph_id is None) != (self.graph_revision is None):
            raise ValueError("graph_id and graph_revision must be provided together")
        return self


class ArtifactSummaryResponse(ApiResponse):
    artifact_id: UUID
    artifact_type: str
    schema_version: int
    content_type: str
    byte_size: int | None = None
    sha256: str | None = None
    text: str | None = None
    content_url: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class RunPortOutputResponse(ApiResponse):
    port: str
    kind: Literal["single", "sequence"]
    value: ArtifactRef | ArtifactRefSequence
    artifacts: list[ArtifactSummaryResponse]


class RunNodeResponse(ApiResponse):
    node_id: str
    status: Literal["succeeded", "failed", "skipped"]
    error: str | None
    outputs: list[RunPortOutputResponse]


class RunResponse(ApiResponse):
    status: Literal["succeeded", "failed"]
    node_runs: list[RunNodeResponse]


class GraphMaterializationsResponse(ApiResponse):
    graph_id: UUID
    graph_revision: int
    node_runs: list[RunNodeResponse]
