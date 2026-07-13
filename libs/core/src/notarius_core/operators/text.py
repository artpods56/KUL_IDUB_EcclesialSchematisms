from typing import Annotated, cast, final, override

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from notarius_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    JsonObject,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.plugins import Plugin


class TextValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: StrictStr


TEXT_VALUE = ArtifactTypeSpec(
    key=ArtifactTypeKey("scalar.text", 1),
    title="Text value",
    payload_schema=cast(JsonObject, TextValue.model_json_schema()),
)

TEXT = Plugin(
    slug="builtin.text",
    title="Text",
)
TEXT.register_artifact_type(TEXT_VALUE)


class TextInputConfig(NodeConfig):
    text: StrictStr = Field(
        description="Multiline text emitted by the node.",
        json_schema_extra={"format": "textarea"},
    )


class TextInputInput(NodeInput):
    pass


class TextInputOutput(NodeOutput):
    text: Annotated[
        TextValue,
        OutPort(TEXT_VALUE),
        Field(description="The configured text value."),
    ]


@TEXT.node(
    operator_id="text.input",
    version=1,
    title="Text input",
)
@final
class TextInputNode(Node[TextInputConfig, TextInputInput, TextInputOutput]):
    """Produces one configured multiline text value."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: TextInputConfig,
        _inputs: TextInputInput,
        /,
    ) -> TextInputOutput:
        return TextInputOutput(text=TextValue(value=config.text))


class SplitTextConfig(NodeConfig):
    separator: StrictStr = Field(
        min_length=1,
        description="Exact text used to separate the input into parts.",
    )


class SplitTextInput(NodeInput):
    text: Annotated[
        TextValue,
        InPort(TEXT_VALUE),
        Field(description="Text value to split."),
    ]


class SplitTextOutput(NodeOutput):
    parts: Annotated[
        list[TextValue],
        OutPort(TEXT_VALUE),
        Field(description="Ordered text parts, including empty parts."),
    ]


@TEXT.node(
    operator_id="text.split",
    version=1,
    title="Split text",
)
@final
class SplitTextNode(Node[SplitTextConfig, SplitTextInput, SplitTextOutput]):
    """Splits text on an exact separator while preserving empty parts."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: SplitTextConfig,
        inputs: SplitTextInput,
        /,
    ) -> SplitTextOutput:
        return SplitTextOutput(
            parts=[
                TextValue(value=part)
                for part in inputs.text.value.split(config.separator)
            ]
        )


class ReplaceTextConfig(NodeConfig):
    search: StrictStr = Field(
        min_length=1,
        description="Exact text to find.",
    )
    replacement: StrictStr = Field(
        default="",
        description="Text substituted for every match.",
    )


class ReplaceTextInput(NodeInput):
    text: Annotated[
        TextValue,
        InPort(TEXT_VALUE),
        Field(description="Text value in which replacements are made."),
    ]


class ReplaceTextOutput(NodeOutput):
    text: Annotated[
        TextValue,
        OutPort(TEXT_VALUE),
        Field(description="Text after all exact replacements."),
    ]


@TEXT.node(
    operator_id="text.replace",
    version=1,
    title="Replace text",
)
@final
class ReplaceTextNode(Node[ReplaceTextConfig, ReplaceTextInput, ReplaceTextOutput]):
    """Replaces every exact occurrence of configured search text."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: ReplaceTextConfig,
        inputs: ReplaceTextInput,
        /,
    ) -> ReplaceTextOutput:
        return ReplaceTextOutput(
            text=TextValue(
                value=inputs.text.value.replace(config.search, config.replacement)
            )
        )


class JoinTextConfig(NodeConfig):
    separator: StrictStr = Field(
        default="",
        description="Text inserted between adjacent parts.",
    )


class JoinTextInput(NodeInput):
    parts: Annotated[
        list[TextValue],
        InPort(TEXT_VALUE),
        Field(description="Ordered text values to join."),
    ]


class JoinTextOutput(NodeOutput):
    text: Annotated[
        TextValue,
        OutPort(TEXT_VALUE),
        Field(description="The joined text value."),
    ]


@TEXT.node(
    operator_id="text.join",
    version=1,
    title="Join text",
)
@final
class JoinTextNode(Node[JoinTextConfig, JoinTextInput, JoinTextOutput]):
    """Joins an ordered text sequence with a configured separator."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: JoinTextConfig,
        inputs: JoinTextInput,
        /,
    ) -> JoinTextOutput:
        return JoinTextOutput(
            text=TextValue(
                value=config.separator.join(part.value for part in inputs.parts)
            )
        )
