"""Conversion-path test scenario: a compound-result plugin plus nodes."""

from typing import Annotated, cast, final, override

from pydantic import BaseModel, ConfigDict, StrictInt

from grafy_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    JsonObject,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from grafy_core.conversions import ArtifactConversion, ArtifactConversionKey
from grafy_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from grafy_core.artifact_contracts import INTEGER_VALUE, TEXT_VALUE
from grafy_core.plugins import Plugin
from grafy_core.runtime.persistence import InlineModelOutputWriter
from grafy_core.runtime.resolvers import InlineModelResolver


class CompoundResultPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    addition: StrictInt
    subtraction: StrictInt


TEST_COMPOUND_RESULT = ArtifactTypeSpec(
    key=ArtifactTypeKey("test.compound_result", 1),
    title="Test compound result",
    payload_schema=cast(JsonObject, CompoundResultPayload.model_json_schema()),
)


def _text_to_compound_result(value: str) -> CompoundResultPayload:
    integer = int(value)
    return CompoundResultPayload(addition=integer + 1, subtraction=integer - 1)


def _failing_text_to_compound_result(value: str) -> CompoundResultPayload:
    raise ValueError(f"Cannot convert {value!r}")


def _invalid_text_to_compound_result(value: str) -> CompoundResultPayload:
    integer = int(value)
    return cast(
        CompoundResultPayload,
        {
            "addition": integer + 1,
            "subtraction": integer - 1,
        },
    )


def _text_to_integer(value: str) -> int:
    return int(value)


TEXT_TO_COMPOUND_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_compound_result", 1),
    source=TEXT_VALUE.key,
    target=TEST_COMPOUND_RESULT.key,
    source_type=str,
    target_type=CompoundResultPayload,
    title="As compound result",
    convert=_text_to_compound_result,
)
FAILING_TEXT_TO_COMPOUND_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_compound_result_failure", 1),
    source=TEXT_VALUE.key,
    target=TEST_COMPOUND_RESULT.key,
    source_type=str,
    target_type=CompoundResultPayload,
    title="Fail as compound result",
    convert=_failing_text_to_compound_result,
)
INVALID_TEXT_TO_COMPOUND_RESULT = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_invalid_compound_result", 1),
    source=TEXT_VALUE.key,
    target=TEST_COMPOUND_RESULT.key,
    source_type=str,
    target_type=CompoundResultPayload,
    title="As invalid compound result",
    convert=_invalid_text_to_compound_result,
)
TEXT_TO_INTEGER = ArtifactConversion(
    key=ArtifactConversionKey("test.scalar.text_to_integer", 1),
    source=TEXT_VALUE.key,
    target=INTEGER_VALUE.key,
    source_type=str,
    target_type=int,
    title="Back to integer",
    convert=_text_to_integer,
)
CONVERSION_PATH_PLUGIN = Plugin(
    slug="test.conversion-path",
    title="Conversion path test plugin",
)
CONVERSION_PATH_PLUGIN.register_artifact_type(TEST_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_type_dependency(INTEGER_VALUE)
CONVERSION_PATH_PLUGIN.register_artifact_type_dependency(TEXT_VALUE)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(TEXT_TO_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(FAILING_TEXT_TO_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(INVALID_TEXT_TO_COMPOUND_RESULT)
CONVERSION_PATH_PLUGIN.register_artifact_conversion(TEXT_TO_INTEGER)
CONVERSION_PATH_PLUGIN.register_resolver(
    lambda context: InlineModelResolver(
        source=TEST_COMPOUND_RESULT.key,
        target=CompoundResultPayload,
        uow=context.uow,
    )
)
CONVERSION_PATH_PLUGIN.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=TEST_COMPOUND_RESULT.key,
        model=CompoundResultPayload,
        uow=context.uow,
    )
)


class CompoundProducerInput(NodeInput):
    left: Annotated[StrictInt, InPort(INTEGER_VALUE)]
    right: Annotated[StrictInt, InPort(INTEGER_VALUE)]


class CompoundProducerOutput(NodeOutput):
    result: Annotated[CompoundResultPayload, OutPort(TEST_COMPOUND_RESULT)]


@CONVERSION_PATH_PLUGIN.node(
    operator_id="test.compound_producer",
    version=1,
    title="Compound producer",
)
@final
class CompoundProducerNode(
    Node[NoConfig, CompoundProducerInput, CompoundProducerOutput]
):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: CompoundProducerInput,
        /,
    ) -> CompoundProducerOutput:
        return CompoundProducerOutput(
            result=CompoundResultPayload(
                addition=inputs.left + inputs.right,
                subtraction=inputs.left - inputs.right,
            )
        )


class CompoundResultConsumerInput(NodeInput):
    result: Annotated[CompoundResultPayload, InPort(TEST_COMPOUND_RESULT)]


class CompoundResultConsumerOutput(NodeOutput):
    value: Annotated[StrictInt, OutPort(INTEGER_VALUE)]


@CONVERSION_PATH_PLUGIN.node(
    operator_id="test.compound_result_consumer",
    version=1,
    title="Compound result consumer",
)
@final
class CompoundResultConsumerNode(
    Node[NoConfig, CompoundResultConsumerInput, CompoundResultConsumerOutput]
):
    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: CompoundResultConsumerInput,
        /,
    ) -> CompoundResultConsumerOutput:
        return CompoundResultConsumerOutput(
            value=inputs.result.addition * inputs.result.subtraction
        )


__all__ = [
    "CONVERSION_PATH_PLUGIN",
    "CompoundProducerInput",
    "CompoundProducerNode",
    "CompoundProducerOutput",
    "CompoundResultConsumerInput",
    "CompoundResultConsumerNode",
    "CompoundResultConsumerOutput",
    "CompoundResultPayload",
    "FAILING_TEXT_TO_COMPOUND_RESULT",
    "INVALID_TEXT_TO_COMPOUND_RESULT",
    "TEST_COMPOUND_RESULT",
    "TEXT_TO_COMPOUND_RESULT",
    "TEXT_TO_INTEGER",
]
