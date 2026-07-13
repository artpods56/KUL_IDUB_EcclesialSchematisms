from typing import cast
from uuid import uuid4

import pytest
from mistralai.client.models import ImageURLChunk, OCRResponse
from pydantic import BaseModel

from notarius_plugin_ocr.mistral import EncodedPageImage, MistralOcrConfig
from notarius_plugin_ocr.mistral_sdk import (
    MistralOcrProviderError,
    MistralSdkOcrProvider,
)


class FakeSdkResponse(BaseModel):
    pages: list[dict[str, object]]
    model: str
    usage_info: dict[str, object]
    document_annotation: str | None = None


class FakeOcrEndpoint:
    def __init__(self) -> None:
        self.document: ImageURLChunk | None = None
        self.model: str | None = None
        self.include_image_base64: bool | None = None
        self.include_blocks: bool | None = None
        self.table_format: str | None = None
        self.timeout_ms: int | None = None

    async def process_async(
        self,
        *,
        model: str,
        document: ImageURLChunk,
        include_image_base64: bool,
        include_blocks: bool,
        table_format: str,
        timeout_ms: int,
    ) -> OCRResponse:
        self.document = document
        self.model = model
        self.include_image_base64 = include_image_base64
        self.include_blocks = include_blocks
        self.table_format = table_format
        self.timeout_ms = timeout_ms
        return cast(
            OCRResponse,
            FakeSdkResponse(
                model="mistral-ocr-4-0",
                usage_info={"pages_processed": 1},
                pages=[
                    {
                        "index": 0,
                        "markdown": "| Name | Value |\n| --- | --- |\n| A | 1 |",
                        "tables": [
                            {
                                "id": "table-0",
                                "content": "| Name | Value |\n| --- | --- |\n| A | 1 |",
                                "format": "markdown",
                            }
                        ],
                        "blocks": [{"type": "table", "bbox": [0, 0, 10, 10]}],
                    }
                ],
            ),
        )


@pytest.mark.asyncio
async def test_mistral_sdk_provider_sends_original_image_and_table_options() -> None:
    endpoint = FakeOcrEndpoint()
    provider = MistralSdkOcrProvider(endpoint)

    response = await provider.process(
        EncodedPageImage(
            artifact_id=uuid4(),
            filename="page.jpg",
            content=b"original-jpeg-bytes",
            content_type="image/jpeg",
        ),
        MistralOcrConfig(model="mistral-ocr-latest", timeout_ms=123_000),
    )

    assert endpoint.model == "mistral-ocr-latest"
    assert endpoint.include_image_base64 is False
    assert endpoint.include_blocks is True
    assert endpoint.table_format == "markdown"
    assert endpoint.timeout_ms == 123_000
    assert endpoint.document is not None
    document_json = endpoint.document.model_dump(mode="json", by_alias=True)
    assert document_json["image_url"] == (
        "data:image/jpeg;base64,b3JpZ2luYWwtanBlZy1ieXRlcw=="
    )
    assert response.model == "mistral-ocr-4-0"
    assert response.pages[0].tables[0].content.endswith("| A | 1 |")
    assert response.pages[0].blocks[0]["type"] == "table"
    assert response.raw_response is not None
    assert "document_annotation" in response.raw_response
    assert response.raw_response["document_annotation"] is None


@pytest.mark.asyncio
async def test_mistral_sdk_provider_requires_server_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    provider = MistralSdkOcrProvider()

    with pytest.raises(MistralOcrProviderError, match="MISTRAL_API_KEY"):
        await provider.process(
            EncodedPageImage(
                artifact_id=uuid4(),
                filename="page.png",
                content=b"png-bytes",
                content_type="image/png",
            ),
            MistralOcrConfig(),
        )
