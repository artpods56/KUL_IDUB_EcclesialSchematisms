from typing import ClassVar, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from notarius_core.prototype.nodes import PortShape


PrototypeNodeGroup = Literal["source", "transform", "ocr", "arithmetic"]
PrototypePortDirection = Literal["input", "output"]


class PrototypeResponse(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(from_attributes=True)


class PrototypeArtifactTypeKeyResponse(PrototypeResponse):
    id: str
    schema_version: int


class PrototypeFieldProjectionResponse(PrototypeResponse):
    path: list[str]
    target_artifact_type: PrototypeArtifactTypeKeyResponse
    title: str


class PrototypeArtifactTypeSpecResponse(PrototypeResponse):
    key: PrototypeArtifactTypeKeyResponse
    title: str
    payload_schema: dict[str, object]
    field_projections: list[PrototypeFieldProjectionResponse]


class PrototypePortResponse(PrototypeResponse):
    name: str
    direction: PrototypePortDirection
    artifact_type: PrototypeArtifactTypeKeyResponse
    shape: PortShape
    variadic: bool = False
    required: bool = True


class PrototypeNodeSpecResponse(PrototypeResponse):
    operator_id: str
    operator_version: int
    title: str
    group: PrototypeNodeGroup
    description: str
    config_schema: dict[str, object]
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    inputs: list[PrototypePortResponse]
    outputs: list[PrototypePortResponse]


class PrototypeNodeRegistryResponse(PrototypeResponse):
    artifact_types: list[PrototypeArtifactTypeSpecResponse]
    nodes: list[PrototypeNodeSpecResponse]


class PrototypeUploadRequest(BaseModel):
    filename: str
    content_base64: str


class PrototypeSampleRequest(BaseModel):
    count: int = Field(default=2, ge=1, le=8)


class PrototypeSelectionItemResponse(PrototypeResponse):
    connector_id: str
    external_uri: str
    display_name: str
    size_bytes: int


class PrototypeRunNodeRequest(BaseModel):
    id: str
    operator_id: str
    config: dict[str, object] = Field(default_factory=dict)


class PrototypeFieldProjectionRequest(BaseModel):
    path: list[str] = Field(min_length=1)


class PrototypeRunEdgeRequest(BaseModel):
    from_node: str
    from_port: str
    to_node: str
    to_port: str
    projection: PrototypeFieldProjectionRequest | None = None


class PrototypeRunRequest(BaseModel):
    nodes: list[PrototypeRunNodeRequest]
    edges: list[PrototypeRunEdgeRequest] = Field(default_factory=list)


class PrototypeArtifactSummaryResponse(PrototypeResponse):
    artifact_id: UUID
    artifact_type: str
    schema_version: int
    content_type: str
    byte_size: int | None = None
    sha256: str | None = None
    text: str | None = None
    content_url: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class PrototypeRunPortOutputResponse(PrototypeResponse):
    port: str
    kind: Literal["single", "sequence"]
    artifacts: list[PrototypeArtifactSummaryResponse]


class PrototypeRunNodeResponse(PrototypeResponse):
    node_id: str
    status: Literal["succeeded", "failed", "skipped"]
    error: str | None
    outputs: list[PrototypeRunPortOutputResponse]


class PrototypeRunResponse(PrototypeResponse):
    status: Literal["succeeded", "failed"]
    node_runs: list[PrototypeRunNodeResponse]
