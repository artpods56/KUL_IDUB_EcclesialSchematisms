from uuid import UUID

import pytest
from pydantic import ValidationError

from notarius_core.artifacts import (
    ArtifactRef,
    InMemoryUnitOfWork,
)
from notarius_core.nodes import NodeExecutionContext, PortShape
from notarius_core.operators.arithmetic import ARITHMETIC, INTEGER_VALUE
from notarius_core.operators.text import (
    INTEGER_TO_TEXT,
    TEXT,
    TEXT_VALUE,
    JoinTextConfig,
    JoinTextInput,
    JoinTextNode,
    ReplaceTextConfig,
    ReplaceTextInput,
    ReplaceTextNode,
    SplitTextConfig,
    SplitTextInput,
    SplitTextNode,
    TextInputConfig,
    TextInputInput,
    TextInputNode,
    TextValue,
    TextValueOutputWriter,
    TextValuePayload,
    TextValueResolver,
)
from notarius_core.plugins import PluginRegistry
from notarius_core.runtime.invocation import (
    InvocationMode,
    map_input_candidates,
    supported_invocation_modes,
)
from notarius_core.runtime.materialization import MaterializationProvenance
from notarius_core.runtime.persistence import ArtifactWriteContext


def test_text_payload_keeps_legacy_public_identity() -> None:
    assert TextValuePayload is TextValue
    assert TextValuePayload.model_json_schema()["title"] == "TextValue"


@pytest.mark.asyncio
async def test_text_input_preserves_multiline_text() -> None:
    output = await TextInputNode().run(
        NodeExecutionContext(node_id="input"),
        TextInputConfig(text="first line\n\nthird line\n"),
        TextInputInput(),
    )

    assert output.text == "first line\n\nthird line\n"
    text_schema = TextInputConfig.model_json_schema()["properties"]["text"]
    assert text_schema["format"] == "textarea"


@pytest.mark.asyncio
async def test_text_split_uses_exact_separator_and_preserves_empty_parts() -> None:
    output = await SplitTextNode().run(
        NodeExecutionContext(node_id="split"),
        SplitTextConfig(separator="||"),
        SplitTextInput(text="||alpha||||beta||"),
    )

    assert output.parts == ["", "alpha", "", "beta", ""]
    assert SplitTextNode.output_contract.ports["parts"].shape is PortShape.MANY


@pytest.mark.asyncio
async def test_text_replace_replaces_every_exact_match() -> None:
    output = await ReplaceTextNode().run(
        NodeExecutionContext(node_id="replace"),
        ReplaceTextConfig(search="cat", replacement="dog"),
        ReplaceTextInput(text="cat scatter cat"),
    )

    assert output.text == "dog sdogter dog"
    assert map_input_candidates(ReplaceTextNode) == ("text",)
    assert supported_invocation_modes(ReplaceTextNode) == (
        InvocationMode.ONCE,
        InvocationMode.MAP,
    )


@pytest.mark.asyncio
async def test_text_join_preserves_order_and_accepts_empty_parts() -> None:
    output = await JoinTextNode().run(
        NodeExecutionContext(node_id="join"),
        JoinTextConfig(separator="|"),
        JoinTextInput(parts=["alpha", "", "beta"]),
    )

    assert output.text == "alpha||beta"
    assert JoinTextNode.input_contract.ports["parts"].shape is PortShape.MANY


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (TextValuePayload, {"value": 1}),
        (TextInputConfig, {"text": 1}),
        (SplitTextInput, {"text": 1}),
        (SplitTextConfig, {"separator": ""}),
        (ReplaceTextConfig, {"search": ""}),
    ],
)
def test_text_models_reject_invalid_values(
    model: type[
        TextValuePayload
        | TextInputConfig
        | SplitTextInput
        | SplitTextConfig
        | ReplaceTextConfig
    ],
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_builtin_integer_to_text_conversion_is_declared_and_nominal() -> None:
    registry = PluginRegistry()
    registry.install(ARITHMETIC)
    registry.install(TEXT)
    registry.freeze()

    assert INTEGER_TO_TEXT.key.id == "builtin.scalar.integer_to_text"
    assert INTEGER_TO_TEXT.key.version == 1
    assert INTEGER_TO_TEXT.source == INTEGER_VALUE.key
    assert INTEGER_TO_TEXT.target == TEXT_VALUE.key
    assert INTEGER_TO_TEXT.source_type is int
    assert INTEGER_TO_TEXT.target_type is str
    assert INTEGER_TO_TEXT.title == "As text"
    assert INTEGER_TO_TEXT.convert(42) == "42"
    assert TEXT.artifact_conversions == (INTEGER_TO_TEXT,)
    assert registry.artifact_conversions == (INTEGER_TO_TEXT,)


@pytest.mark.asyncio
async def test_text_value_adapters_preserve_inline_payload_and_metadata() -> None:
    uow = InMemoryUnitOfWork()
    source_ref = ArtifactRef(
        artifact_id=UUID("00000000-0000-0000-0000-000000000042"),
        artifact_type="scalar.integer",
        schema_version=1,
    )
    writer = TextValueOutputWriter(uow=uow)
    ref = await writer.write(
        "persisted text",
        ArtifactWriteContext(
            node_context=NodeExecutionContext(node_id="text"),
            provenance=MaterializationProvenance(
                refs_by_input={"value": (source_ref,)}
            ),
            metadata={
                "conversion_id": "builtin.scalar.integer_to_text",
                "conversion_version": 1,
            },
        ),
    )
    resolver = TextValueResolver(uow=uow)

    assert await resolver.resolve(ref) == "persisted text"
    async with uow as entered:
        artifact = await entered.artifacts.get(ref.artifact_id)
    assert artifact is not None
    assert artifact.inline_payload == {"value": "persisted text"}
    assert artifact.metadata == {
        "producer_node_id": "text",
        "provenance": {
            "value": [
                {
                    "artifact_id": str(source_ref.artifact_id),
                    "artifact_type": "scalar.integer",
                    "schema_version": 1,
                }
            ]
        },
        "conversion_id": "builtin.scalar.integer_to_text",
        "conversion_version": 1,
    }
