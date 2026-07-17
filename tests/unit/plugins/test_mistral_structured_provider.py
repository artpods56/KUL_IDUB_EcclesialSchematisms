from collections.abc import Sequence
from hashlib import sha256
from io import BytesIO
from uuid import uuid4

import pytest
from mistralai.client.models.assistantmessage import AssistantMessage
from mistralai.client.models.chatcompletionchoice import (
    ChatCompletionChoice,
    ChatCompletionChoiceFinishReason,
)
from mistralai.client.models.chatcompletionresponse import ChatCompletionResponse
from mistralai.client.models.contentchunk import ContentChunk
from mistralai.client.models.imageurlchunk import ImageURLChunk
from mistralai.client.models.jsonschema import JSONSchema
from mistralai.client.models.responseformat import ResponseFormat
from mistralai.client.models.systemmessage import SystemMessage
from mistralai.client.models.textchunk import TextChunk
from mistralai.client.models.usageinfo import UsageInfo
from mistralai.client.models.usermessage import UserMessage

from notarius_core.artifacts import (
    ArtifactObject,
    InMemoryUnitOfWork,
)
from notarius_core.operators.images import RASTER_IMAGE
from notarius_core.operators.prompts import (
    PromptMessage,
    PromptMessageRole,
)
from notarius_core.operators.schemas import parse_json_schema
from notarius_core.ports.storage import SaveFileCommand, StoredFile
from notarius_plugin_llm.mistral import MistralStructuredConfig
from notarius_plugin_llm.mistral_sdk import (
    MISTRAL_MAX_IMAGE_BYTES,
    MISTRAL_MAX_IMAGES,
    MistralStructuredProviderError,
    MistralSdkStructuredProvider,
)


type CapturedSdkMessage = SystemMessage | UserMessage


class TrackingBytesIO(BytesIO):
    def __init__(self, value: bytes) -> None:
        super().__init__(value)
        self.read_sizes: list[int | None] = []

    def read(self, size: int | None = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


class FakeChatEndpoint:
    def __init__(
        self,
        response: ChatCompletionResponse,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.model: str | None = None
        self.messages: list[CapturedSdkMessage] | None = None
        self.temperature: float | None = None
        self.max_tokens: int | None = None
        self.response_format: ResponseFormat | None = None
        self.timeout_ms: int | None = None

    async def complete_async(
        self,
        *,
        model: str,
        messages: list[CapturedSdkMessage],
        temperature: float,
        max_tokens: int,
        response_format: ResponseFormat,
        timeout_ms: int,
    ) -> ChatCompletionResponse:
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.timeout_ms = timeout_ms
        if self._error is not None:
            raise self._error
        return self._response


class FakeStorage:
    def __init__(self) -> None:
        self.files: dict[tuple[str, str], bytes] = {}
        self.load_error: Exception | None = None
        self.last_stream: TrackingBytesIO | None = None

    async def save(self, command: SaveFileCommand) -> StoredFile:
        raise AssertionError(f"Unexpected save to {command.bucket}/{command.path}")

    async def move(
        self,
        bucket: str,
        source_path: str,
        destination_path: str,
    ) -> None:
        raise AssertionError(
            f"Unexpected move in {bucket}: {source_path} to {destination_path}"
        )

    async def load(self, bucket: str, path: str) -> TrackingBytesIO:
        if self.load_error is not None:
            raise self.load_error
        stream = TrackingBytesIO(self.files[(bucket, path)])
        self.last_stream = stream
        return stream

    async def delete(self, bucket: str, path: str) -> None:
        raise AssertionError(f"Unexpected delete from {bucket}/{path}")

    def exists(self, bucket: str, path: str) -> bool:
        return (bucket, path) in self.files


def schema() -> str:
    return (
        '{"type":"object","properties":{"number":{"type":"string"}},'
        '"required":["number"],"additionalProperties":false}'
    )


def mistral_config(
    schema_description: str = "Invoice extraction",
) -> MistralStructuredConfig:
    return MistralStructuredConfig(
        schema_name="invoice",
        schema_description=schema_description,
        strict=True,
    )


def response(
    *,
    content: str | Sequence[ContentChunk] = '{"number":"FV/42"}',
    finish_reason: ChatCompletionChoiceFinishReason = "stop",
    choices: bool = True,
) -> ChatCompletionResponse:
    assistant_content: str | list[ContentChunk]
    if isinstance(content, str):
        assistant_content = content
    else:
        assistant_content = list(content)
    response_choices = []
    if choices:
        response_choices = [
            ChatCompletionChoice(
                index=0,
                finish_reason=finish_reason,
                message=AssistantMessage(content=assistant_content),
            )
        ]
    return ChatCompletionResponse(
        id="completion-1",
        object="chat.completion",
        model="mistral-small-2506",
        usage=UsageInfo(
            prompt_tokens=20,
            completion_tokens=5,
            total_tokens=25,
        ),
        created=123,
        choices=response_choices,
    )


async def add_image(
    uow: InMemoryUnitOfWork,
    storage: FakeStorage,
    *,
    content: bytes = b"image-bytes",
    content_type: str = "image/png",
    byte_size: int | None = None,
    content_hash: str | None = None,
) -> ArtifactObject:
    image = ArtifactObject(
        id=uuid4(),
        artifact_type=RASTER_IMAGE.key.id,
        schema_version=RASTER_IMAGE.key.schema_version,
        content_type=content_type,
        bucket="artifacts",
        object_key="images/page.png",
        byte_size=byte_size,
        sha256=content_hash,
    )
    storage.files[("artifacts", "images/page.png")] = content
    async with uow as transaction:
        await transaction.artifacts.add(image)
        await transaction.commit()
    return image


async def test_sdk_provider_preserves_messages_and_builds_json_schema_request() -> None:
    uow = InMemoryUnitOfWork()
    storage = FakeStorage()
    image = await add_image(uow, storage)
    endpoint = FakeChatEndpoint(response())
    provider = MistralSdkStructuredProvider(
        uow=uow,
        storage=storage,
        endpoint=endpoint,
    )
    config = MistralStructuredConfig(
        model="mistral-small-2506",
        temperature=0.25,
        max_tokens=777,
        timeout_ms=42_000,
        schema_name="invoice",
        schema_description="Invoice extraction",
        strict=True,
    )
    requested_schema = schema()

    result = await provider.complete(
        [
            PromptMessage(
                role=PromptMessageRole.SYSTEM,
                text="Return one invoice.",
            ),
            PromptMessage(
                role=PromptMessageRole.USER,
                text="Read this page.",
                image_refs=[image.ref()],
            ),
            PromptMessage(
                role=PromptMessageRole.SYSTEM,
                text="Use the requested schema.",
            ),
        ],
        requested_schema,
        config,
    )

    assert endpoint.model == "mistral-small-2506"
    assert endpoint.temperature == 0.25
    assert endpoint.max_tokens == 777
    assert endpoint.timeout_ms == 42_000
    assert endpoint.messages is not None
    assert [message.role for message in endpoint.messages] == [
        "system",
        "user",
        "system",
    ]
    assert isinstance(endpoint.messages[0], SystemMessage)
    assert endpoint.messages[0].content == "Return one invoice."
    assert isinstance(endpoint.messages[1], UserMessage)
    user_content = endpoint.messages[1].content
    assert isinstance(user_content, list)
    assert isinstance(user_content[0], TextChunk)
    assert user_content[0].text == "Read this page."
    assert isinstance(user_content[1], ImageURLChunk)
    assert user_content[1].image_url == "data:image/png;base64,aW1hZ2UtYnl0ZXM="
    assert isinstance(endpoint.messages[2], SystemMessage)
    assert endpoint.messages[2].content == "Use the requested schema."

    assert endpoint.response_format is not None
    assert endpoint.response_format.type == "json_schema"
    assert isinstance(endpoint.response_format.json_schema, JSONSchema)
    assert endpoint.response_format.json_schema.name == "invoice"
    assert endpoint.response_format.json_schema.description == "Invoice extraction"
    assert endpoint.response_format.json_schema.strict is True
    assert endpoint.response_format.json_schema.schema_definition == (
        parse_json_schema(requested_schema)
    )
    assert result.value == {"number": "FV/42"}
    assert result.model == "mistral-small-2506"
    assert result.usage == {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
    }
    assert result.raw_response["id"] == "completion-1"


async def test_sdk_provider_omits_empty_schema_description() -> None:
    endpoint = FakeChatEndpoint(response())
    provider = MistralSdkStructuredProvider(
        uow=InMemoryUnitOfWork(),
        storage=FakeStorage(),
        endpoint=endpoint,
    )

    await provider.complete(
        [PromptMessage(role=PromptMessageRole.USER, text="Extract.")],
        schema(),
        mistral_config(schema_description=""),
    )

    assert endpoint.response_format is not None
    assert isinstance(endpoint.response_format.json_schema, JSONSchema)
    assert "description" not in endpoint.response_format.json_schema.model_dump(
        by_alias=True
    )


async def test_sdk_provider_requires_server_api_key_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    provider = MistralSdkStructuredProvider(
        uow=InMemoryUnitOfWork(),
        storage=FakeStorage(),
    )

    with pytest.raises(MistralStructuredProviderError, match="MISTRAL_API_KEY"):
        await provider.complete(
            [PromptMessage(role=PromptMessageRole.USER, text="Extract.")],
            schema(),
            mistral_config(),
        )


@pytest.mark.parametrize(
    ("provider_response", "message"),
    [
        (response(choices=False), "no choices"),
        (response(finish_reason="length"), "finished with reason 'length'"),
        (response(content=[TextChunk(text="not a string")]), "must be a string"),
        (response(content="not-json"), "invalid JSON"),
        (response(content="[]"), "JSON must be an object"),
    ],
)
async def test_sdk_provider_rejects_invalid_completion_shapes(
    provider_response: ChatCompletionResponse,
    message: str,
) -> None:
    provider = MistralSdkStructuredProvider(
        uow=InMemoryUnitOfWork(),
        storage=FakeStorage(),
        endpoint=FakeChatEndpoint(provider_response),
    )

    with pytest.raises(MistralStructuredProviderError, match=message):
        await provider.complete(
            [PromptMessage(role=PromptMessageRole.USER, text="Extract.")],
            schema(),
            mistral_config(),
        )


async def test_sdk_provider_rejects_json_that_does_not_match_schema() -> None:
    provider = MistralSdkStructuredProvider(
        uow=InMemoryUnitOfWork(),
        storage=FakeStorage(),
        endpoint=FakeChatEndpoint(response(content='{"other":"value"}')),
    )

    with pytest.raises(
        MistralStructuredProviderError,
        match="JSON does not match schema 'invoice'",
    ) as captured:
        await provider.complete(
            [PromptMessage(role=PromptMessageRole.USER, text="Extract.")],
            schema(),
            mistral_config(),
        )

    assert captured.value.__cause__ is not None


async def test_sdk_provider_chains_request_and_image_load_failures() -> None:
    request_failure = TimeoutError("endpoint timeout")
    provider = MistralSdkStructuredProvider(
        uow=InMemoryUnitOfWork(),
        storage=FakeStorage(),
        endpoint=FakeChatEndpoint(response(), error=request_failure),
    )
    with pytest.raises(
        MistralStructuredProviderError, match="request failed"
    ) as request:
        await provider.complete(
            [PromptMessage(role=PromptMessageRole.USER, text="Extract.")],
            schema(),
            mistral_config(),
        )
    assert request.value.__cause__ is request_failure

    uow = InMemoryUnitOfWork()
    storage = FakeStorage()
    image = await add_image(uow, storage)
    load_failure = OSError("storage unavailable")
    storage.load_error = load_failure
    provider = MistralSdkStructuredProvider(
        uow=uow,
        storage=storage,
        endpoint=FakeChatEndpoint(response()),
    )
    with pytest.raises(
        MistralStructuredProviderError,
        match=f"load prompt image artifact {image.id}.*artifacts/images/page.png",
    ) as loaded:
        await provider.complete(
            [
                PromptMessage(
                    role=PromptMessageRole.USER,
                    text="Extract.",
                    image_refs=[image.ref()],
                )
            ],
            schema(),
            mistral_config(),
        )
    assert loaded.value.__cause__ is load_failure


async def test_sdk_provider_reports_missing_nested_image() -> None:
    missing_ref = ArtifactObject(
        id=uuid4(),
        artifact_type=RASTER_IMAGE.key.id,
        schema_version=RASTER_IMAGE.key.schema_version,
        content_type="image/png",
    ).ref()
    provider = MistralSdkStructuredProvider(
        uow=InMemoryUnitOfWork(),
        storage=FakeStorage(),
        endpoint=FakeChatEndpoint(response()),
    )

    with pytest.raises(
        MistralStructuredProviderError,
        match=f"Prompt image artifact {missing_ref.artifact_id} was not found",
    ):
        await provider.complete(
            [
                PromptMessage(
                    role=PromptMessageRole.USER,
                    text="Extract.",
                    image_refs=[missing_ref],
                )
            ],
            schema(),
            mistral_config(),
        )


async def test_sdk_provider_enforces_image_count_and_local_image_limits() -> None:
    image_refs = [
        ArtifactObject(
            id=uuid4(),
            artifact_type=RASTER_IMAGE.key.id,
            schema_version=RASTER_IMAGE.key.schema_version,
            content_type="image/png",
        ).ref()
        for _ in range(MISTRAL_MAX_IMAGES + 1)
    ]
    provider = MistralSdkStructuredProvider(
        uow=InMemoryUnitOfWork(),
        storage=FakeStorage(),
        endpoint=FakeChatEndpoint(response()),
    )
    with pytest.raises(MistralStructuredProviderError, match="at most 8 images"):
        await provider.complete(
            [
                PromptMessage(
                    role=PromptMessageRole.USER,
                    text="Extract.",
                    image_refs=image_refs,
                )
            ],
            schema(),
            mistral_config(),
        )

    uow = InMemoryUnitOfWork()
    storage = FakeStorage()
    unsupported = await add_image(
        uow,
        storage,
        content_type="image/tiff",
    )
    provider = MistralSdkStructuredProvider(
        uow=uow,
        storage=storage,
        endpoint=FakeChatEndpoint(response()),
    )
    with pytest.raises(
        MistralStructuredProviderError,
        match=f"{unsupported.id}.*unsupported content type 'image/tiff'",
    ):
        await provider.complete(
            [
                PromptMessage(
                    role=PromptMessageRole.USER,
                    text="Extract.",
                    image_refs=[unsupported.ref()],
                )
            ],
            schema(),
            mistral_config(),
        )

    uow = InMemoryUnitOfWork()
    storage = FakeStorage()
    oversized = await add_image(
        uow,
        storage,
        byte_size=MISTRAL_MAX_IMAGE_BYTES + 1,
    )
    provider = MistralSdkStructuredProvider(
        uow=uow,
        storage=storage,
        endpoint=FakeChatEndpoint(response()),
    )
    with pytest.raises(
        MistralStructuredProviderError,
        match=f"{oversized.id}.*per-image limit",
    ):
        await provider.complete(
            [
                PromptMessage(
                    role=PromptMessageRole.USER,
                    text="Extract.",
                    image_refs=[oversized.ref()],
                )
            ],
            schema(),
            mistral_config(),
        )


async def test_sdk_provider_verifies_recorded_image_digest() -> None:
    uow = InMemoryUnitOfWork()
    storage = FakeStorage()
    image = await add_image(
        uow,
        storage,
        content_hash=sha256(b"original").hexdigest(),
    )
    provider = MistralSdkStructuredProvider(
        uow=uow,
        storage=storage,
        endpoint=FakeChatEndpoint(response()),
    )

    with pytest.raises(
        MistralStructuredProviderError,
        match=f"{image.id} SHA-256 mismatch",
    ):
        await provider.complete(
            [
                PromptMessage(
                    role=PromptMessageRole.USER,
                    text="Extract.",
                    image_refs=[image.ref()],
                )
            ],
            schema(),
            mistral_config(),
        )


async def test_sdk_provider_bounds_reads_without_size_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "notarius_plugin_llm.mistral_sdk.MISTRAL_MAX_IMAGE_BYTES",
        8,
    )
    uow = InMemoryUnitOfWork()
    storage = FakeStorage()
    image = await add_image(
        uow,
        storage,
        content=b"123456789",
    )
    provider = MistralSdkStructuredProvider(
        uow=uow,
        storage=storage,
        endpoint=FakeChatEndpoint(response()),
    )

    with pytest.raises(
        MistralStructuredProviderError,
        match=f"{image.id}.*8-byte per-image limit",
    ):
        await provider.complete(
            [
                PromptMessage(
                    role=PromptMessageRole.USER,
                    text="Extract.",
                    image_refs=[image.ref()],
                )
            ],
            schema(),
            mistral_config(),
        )

    assert storage.last_stream is not None
    assert storage.last_stream.read_sizes == [9]
