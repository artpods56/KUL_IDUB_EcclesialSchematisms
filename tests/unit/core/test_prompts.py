from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from notarius_core.artifacts import (
    ArtifactRef,
    ArtifactRefSequence,
    ArtifactTypeKey,
    InMemoryUnitOfWork,
)
from notarius_core.nodes import NodeExecutionContext, PortShape
from notarius_core.operators.arithmetic import ARITHMETIC
from notarius_core.operators.images import IMAGES, RASTER_IMAGE
from notarius_core.operators.prompts import (
    PROMPTS,
    PROMPT_MESSAGE,
    CreatePromptMessageNode,
    PromptMessage,
    PromptMessageConfig,
    PromptMessageInput,
    PromptMessageRole,
)
from notarius_core.operators.text import TEXT, TEXT_VALUE
from notarius_core.plugins import PluginRegistry, PluginRuntimeContext
from notarius_core.runtime.materialization import MaterializationProvenance
from notarius_core.runtime.persistence import ArtifactWriteContext
from notarius_storage import LocalFileObjectStore


def test_prompt_plugin_declares_fixed_artifacts_nodes_and_port_contracts() -> None:
    registry = PluginRegistry()
    registry.install(IMAGES)
    registry.install(ARITHMETIC)
    registry.install(TEXT)
    registry.install(PROMPTS)
    registry.freeze()

    assert PROMPTS.slug == "builtin.prompt"
    assert PROMPTS.title == "Prompt"
    assert [artifact.key for artifact in PROMPTS.artifact_types] == [
        ArtifactTypeKey("prompt.message", 2),
    ]
    assert [registration.key for registration in PROMPTS.nodes] == [
        ("prompt.message.create", 2),
    ]

    text_input = CreatePromptMessageNode.input_contract.ports["text"]
    assert text_input.accepts == TEXT_VALUE.key
    assert text_input.shape is PortShape.ONE
    assert text_input.target_type is str
    assert text_input.required is True

    image_input = CreatePromptMessageNode.input_contract.ports["images"]
    assert image_input.accepts == RASTER_IMAGE.key
    assert image_input.shape is PortShape.MANY
    assert image_input.preserves_ref_container is True
    assert image_input.allows_none is True
    assert image_input.required is False

    message_output = CreatePromptMessageNode.output_contract.ports["message"]
    assert message_output.produces == PROMPT_MESSAGE.key
    assert message_output.shape is PortShape.ONE

def test_prompt_message_payload_validates_role_text_and_image_refs() -> None:
    image_ref = ArtifactRef.from_key(
        artifact_id=UUID("00000000-0000-0000-0000-000000000001"),
        key=RASTER_IMAGE.key,
    )

    assert PromptMessage(
        role=PromptMessageRole.USER,
        text="Describe the image",
        image_refs=[image_ref],
    ).image_refs == [image_ref]

    with pytest.raises(ValidationError):
        PromptMessage.model_validate({"role": "assistant", "text": "invalid"})
    with pytest.raises(ValidationError):
        PromptMessage.model_validate({"role": "user", "text": 1})
    with pytest.raises(ValidationError, match="System prompt messages cannot include"):
        PromptMessage(
            role=PromptMessageRole.SYSTEM,
            text="System instructions",
            image_refs=[image_ref],
        )
    with pytest.raises(ValidationError, match="must reference image.raster@1"):
        PromptMessage(
            role=PromptMessageRole.USER,
            text="Wrong image type",
            image_refs=[
                ArtifactRef.from_key(
                    artifact_id=UUID("00000000-0000-0000-0000-000000000002"),
                    key=ArtifactTypeKey("example.not_an_image", 1),
                )
            ],
        )


@pytest.mark.asyncio
async def test_message_node_preserves_nested_image_ref_order() -> None:
    first_ref = ArtifactRef.from_key(
        artifact_id=UUID("00000000-0000-0000-0000-000000000011"),
        key=RASTER_IMAGE.key,
    )
    second_ref = ArtifactRef.from_key(
        artifact_id=UUID("00000000-0000-0000-0000-000000000012"),
        key=RASTER_IMAGE.key,
    )
    images = ArtifactRefSequence.from_key(
        key=RASTER_IMAGE.key,
        item_refs=[first_ref, second_ref],
    )

    output = await CreatePromptMessageNode().run(
        NodeExecutionContext(node_id="message"),
        PromptMessageConfig(role=PromptMessageRole.USER),
        PromptMessageInput(text="Read these pages", images=images),
    )

    assert output.message == PromptMessage(
        role=PromptMessageRole.USER,
        text="Read these pages",
        image_refs=[first_ref, second_ref],
    )


@pytest.mark.asyncio
async def test_message_node_rejects_unordered_image_sequence_with_context() -> None:
    images = ArtifactRefSequence(
        artifact_type=RASTER_IMAGE.key.id,
        schema_version=RASTER_IMAGE.key.schema_version,
        item_refs=[],
        ordered=False,
    )

    with pytest.raises(ValueError, match=str(images.sequence_id)):
        await CreatePromptMessageNode().run(
            NodeExecutionContext(node_id="message"),
            PromptMessageConfig(role=PromptMessageRole.USER),
            PromptMessageInput(text="Read these pages", images=images),
        )


@pytest.mark.asyncio
async def test_message_node_forbids_images_on_system_messages() -> None:
    image_ref = ArtifactRef.from_key(
        artifact_id=UUID("00000000-0000-0000-0000-000000000021"),
        key=RASTER_IMAGE.key,
    )

    with pytest.raises(ValidationError, match="System prompt messages cannot include"):
        await CreatePromptMessageNode().run(
            NodeExecutionContext(node_id="system-message"),
            PromptMessageConfig(role=PromptMessageRole.SYSTEM),
            PromptMessageInput(
                text="Follow these rules",
                images=ArtifactRefSequence.from_key(
                    key=RASTER_IMAGE.key,
                    item_refs=[image_ref],
                ),
            ),
        )


@pytest.mark.asyncio
async def test_prompt_inline_factories_round_trip_typed_payloads(
    tmp_path: Path,
) -> None:
    uow = InMemoryUnitOfWork()
    context = PluginRuntimeContext(
        workspace=tmp_path,
        uploads_dir=tmp_path / "uploads",
        storage=LocalFileObjectStore(tmp_path / "objects"),
        uow=uow,
        bucket="artifacts",
    )
    registry = PluginRegistry()
    registry.install(PROMPTS)
    writers = {
        writer.artifact_type: writer for writer in registry.build_writers(context)
    }
    resolvers = {
        resolver.source: resolver for resolver in registry.build_resolvers(context)
    }
    write_context = ArtifactWriteContext(
        node_context=NodeExecutionContext(node_id="prompt"),
        provenance=MaterializationProvenance(refs_by_input={}),
    )
    image_ref = ArtifactRef.from_key(
        artifact_id=UUID("00000000-0000-0000-0000-000000000031"),
        key=RASTER_IMAGE.key,
    )
    message_payload = PromptMessage(
        role=PromptMessageRole.USER,
        text="Extract the invoice",
        image_refs=[image_ref],
    )

    message_ref = await writers[PROMPT_MESSAGE.key].write(
        message_payload,
        write_context,
    )

    assert await resolvers[PROMPT_MESSAGE.key].resolve(message_ref) == message_payload
    async with uow as entered:
        message_artifact = await entered.artifacts.get(message_ref.artifact_id)
    assert message_artifact is not None
    assert message_artifact.inline_payload is not None
    assert message_artifact.inline_payload["image_refs"] == [
        {
            "artifact_id": str(image_ref.artifact_id),
            "artifact_type": RASTER_IMAGE.key.id,
            "schema_version": RASTER_IMAGE.key.schema_version,
            "content_hash": None,
        }
    ]
