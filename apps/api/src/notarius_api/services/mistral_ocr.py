import base64
import os
from typing import Protocol, cast, final, override

from mistralai.client.models import ImageURLChunk, OCRResponse
from mistralai.client.sdk import Mistral

from notarius_core.prototype.mistral_ocr import (
    EncodedPageImage,
    MistralOcrConfig,
    MistralOcrProvider,
    MistralOcrProviderResponse,
)


class MistralOcrSdkPort(Protocol):
    async def process_async(
        self,
        *,
        model: str,
        document: ImageURLChunk,
        include_image_base64: bool,
        include_blocks: bool,
        table_format: str,
        timeout_ms: int,
    ) -> OCRResponse: ...


class MistralOcrProviderError(RuntimeError):
    pass


@final
class MistralSdkOcrProvider(MistralOcrProvider):
    """Mistral SDK adapter; credentials remain outside workflow configuration."""

    def __init__(
        self,
        endpoint: MistralOcrSdkPort | None = None,
    ) -> None:
        self._endpoint = endpoint

    @override
    async def process(
        self,
        image: EncodedPageImage,
        config: MistralOcrConfig,
        /,
    ) -> MistralOcrProviderResponse:
        endpoint = self._endpoint
        if endpoint is None:
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key is None or api_key == "":
                raise MistralOcrProviderError(
                    "MISTRAL_API_KEY is required by the server for Mistral OCR"
                )
            endpoint = cast(MistralOcrSdkPort, Mistral(api_key=api_key).ocr)
            self._endpoint = endpoint

        encoded = base64.b64encode(image.content).decode("ascii")
        document = ImageURLChunk(
            image_url=f"data:{image.content_type};base64,{encoded}"
        )
        try:
            response = await endpoint.process_async(
                model=config.model,
                document=document,
                include_image_base64=False,
                include_blocks=True,
                table_format="markdown",
                timeout_ms=config.timeout_ms,
            )
        except Exception as exc:
            message = (
                f"Mistral OCR request failed for {image.filename!r} with "
                f"model {config.model!r}: {exc.__class__.__name__}"
            )
            raise MistralOcrProviderError(message) from exc

        raw_response = cast(
            dict[str, object],
            response.model_dump(
                mode="json",
                by_alias=True,
            ),
        )
        return MistralOcrProviderResponse.model_validate(
            {
                **raw_response,
                "raw_response": raw_response,
            }
        )
