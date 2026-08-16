from uuid import UUID, uuid4

import pytest

from grafy_core.artifacts import ArtifactRef
from grafy_core.nodes import NodeExecutionContext, PortShape
from grafy_core.operators.images import RASTER_IMAGE
from grafy_core.operators.prompts import (
    PROMPT_MESSAGE,
    PromptMessage,
    PromptMessageRole,
)
from grafy_core.operators.schemas import JSON_SCHEMA
from grafy_plugin_llm.artifacts import STRUCTURED_RESPONSE
from grafy_plugin_llm.mistral import (
    MistralStructuredConfig,
    MistralStructuredExecutionError,
    MistralStructuredInput,
    MistralStructuredNode,
    MistralStructuredProviderResponse,
)


WORKSPACE_ID = UUID("00000000-0000-0000-0000-000000000901")


class FakeStructuredProvider:
    def __init__(
        self,
        response: MistralStructuredProviderResponse,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.messages: list[PromptMessage] | None = None
        self.json_schema: str | None = None
        self.config: MistralStructuredConfig | None = None

    async def complete(
        self,
        messages: list[PromptMessage],
        json_schema: str,
        config: MistralStructuredConfig,
        /,
        *,
        workspace_id: UUID,
    ) -> MistralStructuredProviderResponse:
        self.messages = messages
        self.json_schema = json_schema
        self.config = config
        assert workspace_id == WORKSPACE_ID
        if self._error is not None:
            raise self._error
        return self._response


def structured_schema() -> str:
    return (
        '{"type":"object","properties":{"invoice_number":{"type":"string"}},'
        '"required":["invoice_number"],"additionalProperties":false}'
    )


def structured_config() -> MistralStructuredConfig:
    return MistralStructuredConfig(
        schema_name="invoice",
        schema_description="Extract an invoice number.",
        strict=True,
    )


def test_mistral_structured_node_declares_required_many_and_one_ports() -> None:
    assert MistralStructuredNode.operator_id == "llm.mistral.structured"
    assert MistralStructuredNode.operator_version == 2
    assert MistralStructuredNode.plugin_slug == "external.llm"
    assert MistralStructuredNode.input_contract.ports["messages"].accepts == (
        PROMPT_MESSAGE.key
    )
    assert MistralStructuredNode.input_contract.ports["messages"].shape is (
        PortShape.MANY
    )
    assert MistralStructuredNode.input_contract.ports["messages"].required
    assert (
        MistralStructuredNode.input_contract.ports["json_schema"].accepts
        == JSON_SCHEMA.key
    )
    assert (
        MistralStructuredNode.input_contract.ports["json_schema"].shape is PortShape.ONE
    )
    assert MistralStructuredNode.input_contract.ports["json_schema"].required
    assert MistralStructuredNode.input_contract.ports["json_schema"].target_type is str
    assert MistralStructuredNode.input_contract.ports["json_schema"].title == (
        "JSON Schema"
    )
    assert MistralStructuredNode.output_contract.ports["response"].produces == (
        STRUCTURED_RESPONSE.key
    )


async def test_mistral_structured_node_builds_typed_response_envelope() -> None:
    first_image_id = uuid4()
    second_image_id = uuid4()
    messages = [
        PromptMessage(role=PromptMessageRole.SYSTEM, text="Return JSON."),
        PromptMessage(
            role=PromptMessageRole.USER,
            text="Read both pages.",
            image_refs=[
                ArtifactRef.from_key(
                    artifact_id=first_image_id,
                    key=RASTER_IMAGE.key,
                ),
                ArtifactRef.from_key(
                    artifact_id=second_image_id,
                    key=RASTER_IMAGE.key,
                ),
            ],
        ),
    ]
    schema = structured_schema()
    provider_response = MistralStructuredProviderResponse(
        value={"invoice_number": "FV/42"},
        model="mistral-small-2506",
        usage={"prompt_tokens": 31, "completion_tokens": 7},
        raw_response={"id": "completion-1"},
    )
    provider = FakeStructuredProvider(provider_response)
    node = MistralStructuredNode(provider)
    config = structured_config()

    output = await node.run(
        NodeExecutionContext(workspace_id=WORKSPACE_ID, node_id="structured"),
        config,
        MistralStructuredInput(
            messages=messages,
            json_schema=schema,
        ),
    )

    assert provider.messages == messages
    assert provider.json_schema == schema
    assert provider.config == config
    assert output.response.model_dump(mode="json", by_alias=True) == {
        "value": {"invoice_number": "FV/42"},
        "model": "mistral-small-2506",
        "provider": "mistral",
        "schema": schema,
        "schema_name": "invoice",
        "schema_description": "Extract an invoice number.",
        "schema_strict": True,
        "message_count": 2,
        "source_image_artifact_ids": [
            str(first_image_id),
            str(second_image_id),
        ],
        "usage": {"prompt_tokens": 31, "completion_tokens": 7},
        "raw_response": {"id": "completion-1"},
    }


async def test_mistral_structured_node_chains_contextual_provider_error() -> None:
    failure = TimeoutError("provider timed out")
    node = MistralStructuredNode(
        FakeStructuredProvider(
            MistralStructuredProviderResponse(
                value={},
                model="unused",
                raw_response={},
            ),
            error=failure,
        )
    )

    with pytest.raises(
        MistralStructuredExecutionError,
        match="invoice.*mistral-small-latest.*1 messages.*provider timed out",
    ) as captured:
        await node.run(
            NodeExecutionContext(workspace_id=WORKSPACE_ID),
            structured_config(),
            MistralStructuredInput(
                messages=[
                    PromptMessage(
                        role=PromptMessageRole.USER,
                        text="Extract it.",
                    )
                ],
                json_schema=structured_schema(),
            ),
        )

    assert captured.value.__cause__ is failure


async def test_mistral_structured_node_rejects_provider_schema_mismatch() -> None:
    node = MistralStructuredNode(
        FakeStructuredProvider(
            MistralStructuredProviderResponse(
                value={"wrong": "value"},
                model="mistral-small-latest",
                raw_response={"id": "completion-1"},
            )
        )
    )

    with pytest.raises(
        MistralStructuredExecutionError,
        match="Mistral structured completion failed for schema 'invoice'",
    ) as captured:
        await node.run(
            NodeExecutionContext(workspace_id=WORKSPACE_ID),
            structured_config(),
            MistralStructuredInput(
                messages=[
                    PromptMessage(
                        role=PromptMessageRole.USER,
                        text="Extract it.",
                    )
                ],
                json_schema=structured_schema(),
            ),
        )

    assert captured.value.__cause__ is not None
