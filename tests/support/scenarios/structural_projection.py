"""Structural-projection test scenario: an API-response plugin plus a node."""

from typing import Annotated, cast, final, override

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from grafy_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    JsonObject,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from grafy_core.nodes import Node, NodeExecutionContext, OutPort
from grafy_core.plugins import Plugin
from grafy_core.runtime.persistence import InlineModelOutputWriter


class ApiCustomerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_name: StrictStr = Field(title="Display name")
    retry_count: StrictInt = Field(title="Retry count")


class ApiResponsePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer: ApiCustomerPayload = Field(title="Customer")


API_RESPONSE = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.api_response", 1),
    title="API response",
    payload_schema=cast(JsonObject, ApiResponsePayload.model_json_schema()),
)
STRUCTURAL_PROJECTION_PLUGIN = Plugin(
    slug="test.structural-projection",
    title="Structural projection test plugin",
)
STRUCTURAL_PROJECTION_PLUGIN.register_artifact_type(API_RESPONSE)
STRUCTURAL_PROJECTION_PLUGIN.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=API_RESPONSE.key,
        model=ApiResponsePayload,
        uow=context.uow,
    )
)


class ApiResponseNodeInput(NodeInput):
    pass


class ApiResponseNodeOutput(NodeOutput):
    response: Annotated[ApiResponsePayload, OutPort(API_RESPONSE)]


@STRUCTURAL_PROJECTION_PLUGIN.node(
    operator_id="test.api_response",
    version=1,
    title="API response",
)
@final
class ApiResponseNode(Node[NoConfig, ApiResponseNodeInput, ApiResponseNodeOutput]):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        _inputs: ApiResponseNodeInput,
        /,
    ) -> ApiResponseNodeOutput:
        return ApiResponseNodeOutput(
            response=ApiResponsePayload(
                customer=ApiCustomerPayload(
                    display_name="abc",
                    retry_count=42,
                )
            )
        )


__all__ = [
    "API_RESPONSE",
    "STRUCTURAL_PROJECTION_PLUGIN",
    "ApiCustomerPayload",
    "ApiResponseNode",
    "ApiResponseNodeInput",
    "ApiResponseNodeOutput",
    "ApiResponsePayload",
]
