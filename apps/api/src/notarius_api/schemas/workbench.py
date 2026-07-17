from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from notarius_core.artifacts import ArtifactRef, ArtifactRefSequence, ArtifactTypeKey
from notarius_core.conversions import MAX_ARTIFACT_CONVERSION_HOPS
from notarius_core.nodes import PortShape
from notarius_core.plugins import PluginOrigin


PortDirection = Literal["input", "output"]


class ApiResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)


class ArtifactTypeKeyResponse(ApiResponse):
    id: str
    schema_version: int = Field(ge=1, strict=True)

    def to_key(self) -> ArtifactTypeKey:
        return ArtifactTypeKey(id=self.id, schema_version=self.schema_version)


ArtifactTypeVariableIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class ArtifactTypeBindingModel(ApiResponse):
    model_config: ClassVar[ConfigDict] = ConfigDict(
        from_attributes=True,
        extra="forbid",
    )

    variable: ArtifactTypeVariableIdentifier
    artifact_type: ArtifactTypeKeyResponse


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
    origin: PluginOrigin


class PortResponse(ApiResponse):
    name: str
    title: str | None = None
    description: str | None = None
    direction: PortDirection
    artifact_type: ArtifactTypeKeyResponse | None = None
    artifact_type_variable: ArtifactTypeVariableIdentifier | None = None
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


class NodeSecretInputResponse(ApiResponse):
    name: str
    config_dependencies: list[str]
    title: str
    description: str | None = None


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


class ImageUploadItemResponse(ApiResponse):
    upload_key: str
    filename: str
    byte_size: int


InputPlugIdentifier = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]


class RunInputPlugRequest(BaseModel):
    id: InputPlugIdentifier
    port: InputPlugIdentifier


class RunNodeRequest(BaseModel):
    id: str
    operator_id: str
    operator_version: int
    config: dict[str, object] = Field(default_factory=dict)
    input_plugs: list[RunInputPlugRequest] = Field(default_factory=list)
    artifact_type_bindings: list[ArtifactTypeBindingModel] = Field(
        default_factory=list,
    )

    @model_validator(mode="after")
    def validate_artifact_type_bindings(self) -> Self:
        variables = [binding.variable for binding in self.artifact_type_bindings]
        if len(variables) != len(set(variables)):
            raise ValueError("Node artifact type binding variables must be unique")
        return self


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
    to_plug: InputPlugIdentifier | None = None
    projection: FieldProjectionRequest | None = None
    conversion_path: list[ArtifactConversionRequest] = Field(
        default_factory=list,
        max_length=MAX_ARTIFACT_CONVERSION_HOPS,
    )
    collection_mode: Literal["direct", "map"] = "direct"

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
                "Run edge cannot declare both conversion and conversion_path"
            )
        normalized = dict(raw)
        conversion = normalized.pop("conversion")
        normalized["conversion_path"] = [] if conversion is None else [conversion]
        return normalized


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
    secret_graph_id: UUID | None = None
    secret_graph_revision: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_graph_context(self) -> Self:
        if (self.graph_id is None) != (self.graph_revision is None):
            raise ValueError("graph_id and graph_revision must be provided together")
        if (self.secret_graph_id is None) != (self.secret_graph_revision is None):
            raise ValueError(
                "secret_graph_id and secret_graph_revision must be provided together"
            )
        if (
            self.graph_id is not None
            and self.secret_graph_id is not None
            and (
                self.graph_id != self.secret_graph_id
                or self.graph_revision != self.secret_graph_revision
            )
        ):
            raise ValueError(
                "graph and secret graph contexts must identify the same saved "
                "graph revision when both are provided"
            )
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


class RunExecutionResponse(ApiResponse):
    execution_id: UUID
    status: Literal[
        "queued",
        "running",
        "cancelling",
        "cancelled",
        "succeeded",
        "failed",
    ]
    active_node_id: str | None
    result: RunResponse | None
    error: str | None


class GraphMaterializationsResponse(ApiResponse):
    graph_id: UUID
    graph_revision: int
    node_runs: list[RunNodeResponse]
