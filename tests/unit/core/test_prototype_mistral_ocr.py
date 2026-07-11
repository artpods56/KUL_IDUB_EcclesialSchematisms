from uuid import uuid4

import pytest

from notarius_core.prototype.mistral_ocr import (
    EncodedPageImage,
    MistralOcrConfig,
    MistralOcrExecutionError,
    MistralOcrInput,
    MistralOcrNode,
    MistralOcrProviderResponse,
)
from notarius_core.prototype.nodes import NodeExecutionContext


class FakeMistralProvider:
    async def process(
        self,
        image: EncodedPageImage,
        config: MistralOcrConfig,
        /,
    ) -> MistralOcrProviderResponse:
        return MistralOcrProviderResponse.model_validate(
            {
                "model": "mistral-ocr-4-0",
                "usage_info": {"pages_processed": 1},
                "raw_response": {
                    "model": "mistral-ocr-4-0",
                    "usage_info": {"pages_processed": 1},
                    "document_annotation": None,
                },
                "pages": [
                    {
                        "index": 0,
                        "markdown": f"# {image.filename}",
                        "tables": [
                            {
                                "id": "table-0",
                                "content": "| A | B |\n| --- | --- |\n| 1 | 2 |",
                                "format": "markdown",
                            }
                        ],
                        "blocks": [{"type": "title", "text": image.filename}],
                    }
                ],
            }
        )


class FailingMistralProvider:
    async def process(
        self,
        image: EncodedPageImage,
        config: MistralOcrConfig,
        /,
    ) -> MistralOcrProviderResponse:
        del image, config
        raise RuntimeError("provider unavailable")


@pytest.mark.asyncio
async def test_mistral_node_preserves_full_response_and_source_identity() -> None:
    artifact_id = uuid4()
    node = MistralOcrNode(FakeMistralProvider())

    output = await node.run(
        NodeExecutionContext(node_id="mistral"),
        MistralOcrConfig(),
        MistralOcrInput(
            pages=[
                EncodedPageImage(
                    artifact_id=artifact_id,
                    filename="page-01.jpg",
                    content=b"jpeg",
                    content_type="image/jpeg",
                )
            ]
        ),
    )

    result = output.responses[0]
    assert result.source_image_artifact_id == artifact_id
    assert result.source_image == "page-01.jpg"
    assert result.sequence_index == 0
    assert result.engine == "mistral.ocr"
    assert result.model == "mistral-ocr-4-0"
    assert result.markdown == "<!-- page 0 -->\n# page-01.jpg\n"
    assert result.pages[0].blocks == [
        {"type": "title", "text": "page-01.jpg"}
    ]
    assert result.raw_response["usage_info"] == {"pages_processed": 1}
    assert "document_annotation" in result.raw_response
    assert result.raw_response["document_annotation"] is None


@pytest.mark.asyncio
async def test_mistral_node_error_names_source_and_model() -> None:
    node = MistralOcrNode(FailingMistralProvider())

    with pytest.raises(
        MistralOcrExecutionError,
        match="page-01.jpg.*mistral-ocr-latest",
    ):
        await node.run(
            NodeExecutionContext(node_id="mistral"),
            MistralOcrConfig(),
            MistralOcrInput(
                pages=[
                    EncodedPageImage(
                        artifact_id=uuid4(),
                        filename="page-01.jpg",
                        content=b"jpeg",
                        content_type="image/jpeg",
                    )
                ]
            ),
        )
