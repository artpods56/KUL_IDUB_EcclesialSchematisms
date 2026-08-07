from enum import StrEnum
from typing import Annotated, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    model_validator,
)

from notarius_core.artifacts import (
    Artifact,
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    ArtifactTypeSpec,
    JsonObject,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import InPort, OutPort
from notarius_core.operators.images import RASTER_IMAGE
from notarius_core.operators.text import TEXT_VALUE
from notarius_core.plugins import NodeCachePolicy, Plugin
from notarius_core.runtime.persistence import InlineModelOutputWriter
from notarius_core.runtime.resolvers import InlineModelResolver


class PromptMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"


class PromptMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: PromptMessageRole
    text: StrictStr
    image_refs: list[ArtifactRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_image_refs(self) -> Self:
        for index, image_ref in enumerate(self.image_refs):
            if image_ref.key() != RASTER_IMAGE.key:
                raise ValueError(
                    f"image_refs[{index}] must reference {RASTER_IMAGE.key.id}@"
                    f"{RASTER_IMAGE.key.schema_version}, got {image_ref.artifact_type}@"
                    f"{image_ref.schema_version}"
                )
        if self.role is PromptMessageRole.SYSTEM and self.image_refs:
            raise ValueError("System prompt messages cannot include images")
        return self


PROMPT_MESSAGE = ArtifactTypeSpec(
    key=ArtifactTypeKey("prompt.message", 2),
    title="Prompt message",
    payload_schema=cast(JsonObject, PromptMessage.model_json_schema()),
)


PROMPTS = Plugin(
    slug="builtin.prompt",
    title="Prompt",
)
class PromptMessageConfig(NodeConfig):
    role: PromptMessageRole = Field(
        default=PromptMessageRole.USER,
        description="Role assigned to the prompt message.",
    )


class PromptMessageInput(NodeInput):
    text: Annotated[
        StrictStr,
        InPort(TEXT_VALUE),
        Field(description="Text content of the prompt message."),
    ]
    images: Annotated[
        ArtifactRefSequence | None,
        InPort(RASTER_IMAGE),
        Field(description="Optional ordered image artifacts for a user message."),
    ] = None


class PromptMessageOutput(NodeOutput):
    message: Annotated[
        PromptMessage,
        OutPort(PROMPT_MESSAGE),
        Field(description="The composed prompt message."),
    ]


@PROMPTS.function_node(
    operator_id="prompt.message.create",
    version=2,
    title="Create prompt message",
    cache_policy=NodeCachePolicy.EXACT,
)
async def create_prompt_message(
    config: PromptMessageConfig,
    inputs: PromptMessageInput,
) -> PromptMessageOutput:
    """Composes prompt text and optional image references into one message."""
    if inputs.images is not None and not inputs.images.ordered:
        raise ValueError(
            "Cannot create a prompt message from unordered image sequence "
            f"{inputs.images.sequence_id}"
        )
    image_refs = [] if inputs.images is None else inputs.images.item_refs
    return PromptMessageOutput(
        message=PromptMessage(
            role=config.role,
            text=inputs.text,
            image_refs=image_refs,
        )
    )


CreatePromptMessageNode = PROMPTS.nodes[-1].node_class


PROMPTS.register(
    Artifact(
        spec=PROMPT_MESSAGE,
        resolver=lambda context: InlineModelResolver(
            source=PROMPT_MESSAGE.key,
            target=PromptMessage,
            uow=context.uow,
        ),
        writer=lambda context: InlineModelOutputWriter(
            artifact_type=PROMPT_MESSAGE.key,
            model=PromptMessage,
            uow=context.uow,
        ),
    )
)


__all__ = [
    "PROMPTS",
    "PROMPT_MESSAGE",
    "CreatePromptMessageNode",
    "PromptMessage",
    "PromptMessageConfig",
    "PromptMessageInput",
    "PromptMessageOutput",
    "PromptMessageRole",
]
