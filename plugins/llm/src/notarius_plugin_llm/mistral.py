from typing import Annotated, Protocol, final, override
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from notarius_core.artifacts import JsonObject, NodeConfig, NodeInput, NodeOutput
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.operators.prompts import (
    PROMPT_MESSAGE,
    PromptMessage,
)
from notarius_core.operators.schemas import JSON_SCHEMA, validate_json_schema_value
from notarius_core.plugins import PluginRuntimeContext

from notarius_plugin_llm.artifacts import (
    STRUCTURED_RESPONSE,
    StructuredResponsePayload,
)
from notarius_plugin_llm.declaration import LLM


class MistralStructuredConfig(NodeConfig):
    model: str = Field(
        default="mistral-small-latest",
        min_length=1,
        description="Mistral chat model identifier.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature passed to Mistral.",
    )
    max_tokens: int = Field(
        default=2_048,
        ge=1,
        le=131_072,
        description="Maximum number of completion tokens.",
    )
    timeout_ms: int = Field(
        default=120_000,
        ge=1_000,
        le=900_000,
        description="Maximum provider request time in milliseconds.",
    )
    schema_name: StrictStr = Field(
        default="structured_response",
        min_length=1,
        description="Provider-facing name for the response schema.",
    )
    schema_description: StrictStr = Field(
        default="",
        description="Optional provider-facing description of the response schema.",
    )
    strict: bool = Field(
        default=True,
        description="Whether Mistral must enforce the response schema strictly.",
    )


class MistralStructuredInput(NodeInput):
    messages: Annotated[
        list[PromptMessage],
        InPort(PROMPT_MESSAGE),
        Field(min_length=1, description="Ordered prompt messages."),
    ]
    json_schema: Annotated[
        StrictStr,
        InPort(JSON_SCHEMA),
        Field(
            title="JSON Schema",
            description="JSON Schema required for the provider response.",
        ),
    ]


class MistralStructuredOutput(NodeOutput):
    response: Annotated[
        StructuredResponsePayload,
        OutPort(STRUCTURED_RESPONSE),
        Field(description="Validated structured completion and provider metadata."),
    ]


class MistralStructuredProviderResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: JsonObject
    model: str = Field(min_length=1)
    usage: JsonObject = Field(default_factory=dict)
    raw_response: JsonObject


class MistralStructuredProvider(Protocol):
    async def complete(
        self,
        messages: list[PromptMessage],
        json_schema: str,
        config: MistralStructuredConfig,
        /,
        *,
        workspace_id: UUID,
    ) -> MistralStructuredProviderResponse: ...


class MistralStructuredExecutionError(RuntimeError):
    pass


def build_mistral_structured_node(
    context: PluginRuntimeContext,
) -> "MistralStructuredNode":
    from notarius_plugin_llm.mistral_sdk import MistralSdkStructuredProvider

    return MistralStructuredNode(
        MistralSdkStructuredProvider(
            uow=context.uow,
            storage=context.storage,
        )
    )


@LLM.node(
    operator_id="llm.mistral.structured",
    version=2,
    title="Mistral Structured Output",
    factory=build_mistral_structured_node,
)
@final
class MistralStructuredNode(
    Node[
        MistralStructuredConfig,
        MistralStructuredInput,
        MistralStructuredOutput,
    ]
):
    """Produces one JSON object constrained by an explicit schema."""

    def __init__(self, provider: MistralStructuredProvider) -> None:
        self._provider = provider

    @override
    async def run(
        self,
        context: NodeExecutionContext,
        config: MistralStructuredConfig,
        inputs: MistralStructuredInput,
        /,
    ) -> MistralStructuredOutput:
        try:
            completion = await self._provider.complete(
                inputs.messages,
                inputs.json_schema,
                config,
                workspace_id=context.workspace_id,
            )
            validated_value = validate_json_schema_value(
                inputs.json_schema,
                completion.value,
            )
        except Exception as exc:
            message = (
                f"Mistral structured completion failed for schema "
                f"{config.schema_name!r} with model "
                f"{config.model!r} and {len(inputs.messages)} messages: {exc}"
            )
            raise MistralStructuredExecutionError(message) from exc

        return MistralStructuredOutput(
            response=StructuredResponsePayload(
                value=validated_value,
                model=completion.model,
                schema=inputs.json_schema,
                schema_name=config.schema_name,
                schema_description=config.schema_description,
                schema_strict=config.strict,
                message_count=len(inputs.messages),
                source_image_artifact_ids=[
                    ref.artifact_id
                    for prompt_message in inputs.messages
                    for ref in prompt_message.image_refs
                ],
                usage=completion.usage,
                raw_response=completion.raw_response,
            )
        )
