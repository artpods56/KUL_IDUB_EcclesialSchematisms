import json
import os
from typing import Any, Final, Protocol, cast, final, override

from mistralai.client.models.chatcompletionresponse import ChatCompletionResponse
from mistralai.client.models.contentchunk import ContentChunk
from mistralai.client.models.imageurlchunk import ImageURLChunk
from mistralai.client.models.jsonschema import JSONSchema
from mistralai.client.models.responseformat import ResponseFormat
from mistralai.client.models.systemmessage import SystemMessage
from mistralai.client.models.textchunk import TextChunk
from mistralai.client.models.usermessage import UserMessage
from mistralai.client.sdk import Mistral

from notarius_core.artifacts import JsonObject, UnitOfWorkPort
from notarius_core.operators.prompts import (
    PromptMessage,
    PromptMessageRole,
)
from notarius_core.operators.schemas import (
    parse_json_schema,
    validate_json_schema_value,
)
from notarius_core.ports.storage import FileStoragePort

from notarius_plugin_llm.mistral import (
    MistralStructuredConfig,
    MistralStructuredProvider,
    MistralStructuredProviderResponse,
)
from notarius_plugin_llm.prompt_images import (
    PromptImageDataError,
    PromptImageDataLoader,
)


type MistralSdkMessage = SystemMessage | UserMessage

MISTRAL_MAX_IMAGES: Final = 8
MISTRAL_MAX_IMAGE_BYTES: Final = 20_000_000
MISTRAL_MAX_TOTAL_IMAGE_BYTES: Final = 50_000_000
MISTRAL_SUPPORTED_IMAGE_CONTENT_TYPES: Final = frozenset(
    {
        "image/gif",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)


class MistralChatEndpoint(Protocol):
    async def complete_async(
        self,
        *,
        model: str,
        messages: list[MistralSdkMessage],
        temperature: float,
        max_tokens: int,
        response_format: ResponseFormat,
        timeout_ms: int,
    ) -> ChatCompletionResponse: ...


class MistralStructuredProviderError(RuntimeError):
    pass


@final
class MistralSdkStructuredProvider(MistralStructuredProvider):
    """Mistral SDK adapter with server-owned credentials and image loading."""

    def __init__(
        self,
        *,
        uow: UnitOfWorkPort,
        storage: FileStoragePort,
        endpoint: MistralChatEndpoint | None = None,
    ) -> None:
        self._uow = uow
        self._storage = storage
        self._endpoint = endpoint

    @override
    async def complete(
        self,
        messages: list[PromptMessage],
        json_schema: str,
        config: MistralStructuredConfig,
        /,
    ) -> MistralStructuredProviderResponse:
        schema_definition = parse_json_schema(json_schema)
        sdk_messages = await self._sdk_messages(messages)
        endpoint = self._endpoint
        if endpoint is None:
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key is None or api_key == "":
                raise MistralStructuredProviderError(
                    "MISTRAL_API_KEY is required by the server for Mistral "
                    "structured output"
                )
            endpoint = cast(MistralChatEndpoint, Mistral(api_key=api_key).chat)
            self._endpoint = endpoint

        sdk_json_schema = JSONSchema(
            name=config.schema_name,
            strict=config.strict,
            schema_definition=cast(
                dict[str, Any],
                schema_definition,
            ),
        )
        if config.schema_description != "":
            sdk_json_schema = JSONSchema(
                name=config.schema_name,
                description=config.schema_description,
                strict=config.strict,
                schema_definition=cast(
                    dict[str, Any],
                    schema_definition,
                ),
            )
        response_format = ResponseFormat(
            type="json_schema",
            json_schema=sdk_json_schema,
        )
        try:
            response = await endpoint.complete_async(
                model=config.model,
                messages=sdk_messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format=response_format,
                timeout_ms=config.timeout_ms,
            )
        except Exception as exc:
            message = (
                f"Mistral structured request failed for schema "
                f"{config.schema_name!r} "
                f"with model {config.model!r}: {exc.__class__.__name__}"
            )
            raise MistralStructuredProviderError(message) from exc

        if not response.choices:
            raise MistralStructuredProviderError(
                f"Mistral returned no choices for schema {config.schema_name!r} with "
                f"model {config.model!r}"
            )

        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise MistralStructuredProviderError(
                f"Mistral choice {choice.index} finished with reason "
                f"{choice.finish_reason!r} for schema {config.schema_name!r}"
            )
        if choice.message is None:
            raise MistralStructuredProviderError(
                f"Mistral choice {choice.index} did not contain a message for "
                f"schema {config.schema_name!r}"
            )

        content = choice.message.content
        if not isinstance(content, str):
            raise MistralStructuredProviderError(
                f"Mistral choice {choice.index} content must be a string for "
                f"schema {config.schema_name!r}, got {type(content).__name__}"
            )
        try:
            decoded: object = json.loads(content)
        except json.JSONDecodeError as exc:
            message = (
                f"Mistral choice {choice.index} returned invalid JSON for schema "
                f"{config.schema_name!r}"
            )
            raise MistralStructuredProviderError(message) from exc
        if not isinstance(decoded, dict):
            raise MistralStructuredProviderError(
                f"Mistral choice {choice.index} JSON must be an object for schema "
                f"{config.schema_name!r}, got {type(decoded).__name__}"
            )
        try:
            value = validate_json_schema_value(
                json_schema,
                cast(JsonObject, decoded),
            )
        except Exception as exc:
            message = (
                f"Mistral choice {choice.index} JSON does not match schema "
                f"{config.schema_name!r}"
            )
            raise MistralStructuredProviderError(message) from exc

        return MistralStructuredProviderResponse(
            value=value,
            model=response.model,
            usage=cast(
                JsonObject,
                response.usage.model_dump(mode="json", by_alias=True),
            ),
            raw_response=cast(
                JsonObject,
                response.model_dump(mode="json", by_alias=True),
            ),
        )

    async def _sdk_messages(
        self,
        messages: list[PromptMessage],
    ) -> list[MistralSdkMessage]:
        image_count = sum(len(message.image_refs) for message in messages)
        if image_count > MISTRAL_MAX_IMAGES:
            raise MistralStructuredProviderError(
                f"Mistral structured requests support at most "
                f"{MISTRAL_MAX_IMAGES} images, got {image_count}"
            )

        sdk_messages: list[MistralSdkMessage] = []
        total_image_bytes = 0
        image_loader = PromptImageDataLoader(
            uow=self._uow,
            storage=self._storage,
            provider_title="Mistral",
            max_image_bytes=MISTRAL_MAX_IMAGE_BYTES,
            max_total_image_bytes=MISTRAL_MAX_TOTAL_IMAGE_BYTES,
            supported_content_types=MISTRAL_SUPPORTED_IMAGE_CONTENT_TYPES,
        )
        for index, message in enumerate(messages):
            if message.role == PromptMessageRole.SYSTEM:
                if message.image_refs:
                    raise MistralStructuredProviderError(
                        f"System prompt message {index} cannot contain image refs"
                    )
                sdk_messages.append(SystemMessage(content=message.text))
                continue

            if message.role != PromptMessageRole.USER:
                raise MistralStructuredProviderError(
                    f"Unsupported prompt message role {message.role!r} at "
                    f"position {index}"
                )
            if not message.image_refs:
                sdk_messages.append(UserMessage(content=message.text))
                continue

            content: list[ContentChunk] = [TextChunk(text=message.text)]
            for image_ref in message.image_refs:
                try:
                    image_url, image_bytes = await image_loader.data_url(
                        image_ref,
                        remaining_total_bytes=(
                            MISTRAL_MAX_TOTAL_IMAGE_BYTES - total_image_bytes
                        ),
                    )
                except PromptImageDataError as exc:
                    cause = exc.__cause__ if exc.__cause__ is not None else exc
                    raise MistralStructuredProviderError(str(exc)) from cause
                total_image_bytes += image_bytes
                content.append(
                    ImageURLChunk(
                        image_url=image_url,
                    )
                )
            sdk_messages.append(UserMessage(content=content))
        return sdk_messages
