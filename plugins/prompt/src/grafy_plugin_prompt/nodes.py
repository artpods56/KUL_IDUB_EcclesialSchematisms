from typing import Annotated

from pydantic import Field, StrictStr

from grafy_core.artifact_contracts import RASTER_IMAGE, TEXT_VALUE
from grafy_core.artifacts import (
    Artifact,
    ArtifactRefSequence,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from grafy_core.nodes import InPort, OutPort
from grafy_core.plugins import NodeCachePolicy
from grafy_core.prompt_contracts import (
    PROMPT_MESSAGE,
    PromptMessage,
    PromptMessageRole,
)
from grafy_core.runtime.persistence import InlineModelOutputWriter
from grafy_core.runtime.resolvers import InlineModelResolver

from grafy_plugin_prompt.declaration import PROMPTS


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
    "CreatePromptMessageNode",
    "PromptMessageConfig",
    "PromptMessageInput",
    "PromptMessageOutput",
]
