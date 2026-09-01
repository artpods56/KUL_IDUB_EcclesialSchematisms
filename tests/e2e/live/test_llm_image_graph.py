from base64 import b64decode
import os
from uuid import UUID

from pydantic import SecretStr
import pytest

from grafy_client import GrafyClient, GraphBuilder
from grafy_core.artifacts import NoConfig
from grafy_plugin_image.nodes import (
    ImageUploadConfig,
    ImageUploadItem,
    UploadImagesNode,
)
from grafy_plugin_llm.artifacts import CompletionPayload
from grafy_plugin_llm.openai_compatible import (
    OpenAICompatibleConfig,
    OpenAICompatibleNode,
)
from grafy_plugin_prompt.nodes import CreatePromptMessageNode, PromptMessageConfig
from grafy_plugin_sequence.nodes import CollectNode
from grafy_plugin_text.nodes import TextInputConfig, TextInputNode


E2E_WORKSPACE_ID = UUID("00000000-0000-4000-8000-000000000002")
ONE_PIXEL_PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


@pytest.mark.asyncio
async def test_saved_multimodal_graph_executes_through_live_http_api() -> None:
    base_url = os.environ.get("GRAFY_E2E_BASE_URL")
    token = os.environ.get("GRAFY_E2E_TOKEN")
    if base_url is None or token is None:
        pytest.skip(
            "run scripts/e2e/run_live.py to provide a disposable live deployment"
        )

    async with GrafyClient(
        base_url=base_url,
        token=SecretStr(token),
        timeout=30,
    ) as client:
        upload = await client.uploads.create(
            E2E_WORKSPACE_ID,
            filename="one-pixel.png",
            content=ONE_PIXEL_PNG,
            content_type="image/png",
        )
        catalog = await client.catalog.get(E2E_WORKSPACE_ID)

        builder = GraphBuilder(catalog)
        image = builder.add(
            UploadImagesNode,
            ImageUploadConfig(
                uploads=[
                    ImageUploadItem(
                        upload_key=upload.upload_key,
                        filename=upload.filename,
                        byte_size=upload.byte_size,
                    )
                ]
            ),
        )
        text = builder.add(
            TextInputNode,
            TextInputConfig(text="Describe the supplied image."),
        )
        prompt = builder.add(CreatePromptMessageNode, PromptMessageConfig())
        messages = builder.add(CollectNode, NoConfig())
        completion = builder.add(
            OpenAICompatibleNode,
            OpenAICompatibleConfig(
                base_url="https://grafy-e2e-provider:18443/v1",
                model="vision-e2e",
                timeout_ms=10_000,
                max_retries=0,
            ),
        )
        builder.connect(text.output("text"), prompt.input("text"))
        builder.connect(image.output("images"), prompt.input("images"))
        builder.connect(prompt.output("message"), messages.input("items"))
        builder.connect(messages.output("items"), completion.input("messages"))

        saved = await client.graphs.create(
            E2E_WORKSPACE_ID,
            name="Live multimodal E2E",
            document=builder.build(),
        )
        secret = await client.graphs.configure_secret(
            E2E_WORKSPACE_ID,
            saved.id,
            node_id=completion.node_id,
            secret_name="api_key",
            value=SecretStr("grafy-e2e-provider-key"),
            expected_revision=saved.revision,
        )
        assert secret.configured is True

        execution = await client.graphs.execute(
            E2E_WORKSPACE_ID,
            saved.id,
            expected_revision=saved.revision,
            idempotency_key="live-multimodal-e2e-v1",
        )
        terminal = await execution.wait(timeout=120, poll_interval=0.2)

    assert terminal.status == "succeeded", terminal.error
    assert terminal.result is not None
    assert terminal.result.status == "succeeded"
    assert {node.status for node in terminal.result.node_runs} == {"succeeded"}
    output = terminal.result.node(completion.node_id).output("completion")
    assert output.artifact_type == "llm.completion"
    assert output.artifact_id.int != 0
    completion_artifact = output.artifacts[0]
    assert completion_artifact.text is not None
    completion_payload = CompletionPayload.model_validate_json(
        completion_artifact.text
    )
    assert completion_payload.content == "The request contained text and one image."
    image_output = terminal.result.node(image.node_id).output("images")
    assert len(image_output.artifacts) == 1
    assert completion_payload.source_image_artifact_ids == [
        image_output.artifacts[0].artifact_id
    ]
