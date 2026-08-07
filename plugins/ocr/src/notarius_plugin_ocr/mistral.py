from typing import Annotated, Literal, Protocol, cast, final, override
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from notarius_core.artifacts import (
    JsonObject,
    NodeConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.operators.images import RASTER_IMAGE

from notarius_plugin_ocr.artifacts import MISTRAL_OCR_RESPONSE
from notarius_plugin_ocr.declaration import OCR


class EncodedPageImage(BaseModel):
    """Original source bytes delivered to providers that accept encoded images."""

    artifact_id: UUID
    filename: str
    content: bytes
    content_type: str


class MistralOcrTablePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    content: str
    format: str = "markdown"
    word_confidence_scores: list[JsonObject] | None = None


class MistralOcrPagePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = Field(ge=0)
    markdown: str
    images: list[JsonObject] = Field(default_factory=list)
    dimensions: JsonObject | None = None
    tables: list[MistralOcrTablePayload] = Field(default_factory=list)
    hyperlinks: list[str] = Field(default_factory=list)
    header: str | None = None
    footer: str | None = None
    confidence_scores: JsonObject | None = None
    blocks: list[JsonObject] = Field(default_factory=list)


class MistralOcrProviderResponse(BaseModel):
    """Typed subset of the provider response while preserving unknown fields."""

    model_config = ConfigDict(extra="allow")

    pages: list[MistralOcrPagePayload]
    model: str
    usage_info: JsonObject = Field(default_factory=dict)
    document_annotation: str | None = None
    raw_response: JsonObject | None = Field(default=None, exclude=True)


class MistralOcrResponsePayload(BaseModel):
    source_image_artifact_id: UUID
    source_image: str
    sequence_index: int = Field(ge=0)
    engine: Literal["mistral.ocr"] = "mistral.ocr"
    model: str
    markdown: str
    pages: list[MistralOcrPagePayload]
    usage_info: JsonObject = Field(default_factory=dict)
    raw_response: JsonObject


class MistralOcrConfig(NodeConfig):
    model: str = Field(
        default="mistral-ocr-latest",
        min_length=1,
        description="Mistral OCR model identifier.",
    )
    timeout_ms: int = Field(
        default=300_000,
        ge=1_000,
        le=900_000,
        description="Maximum provider request time in milliseconds.",
    )


class MistralOcrInput(NodeInput):
    pages: Annotated[
        list[EncodedPageImage],
        InPort(RASTER_IMAGE),
        Field(min_length=1, description="Ordered encoded raster images to process."),
    ]


class MistralOcrOutput(NodeOutput):
    responses: Annotated[
        list[MistralOcrResponsePayload],
        OutPort(MISTRAL_OCR_RESPONSE),
        Field(description="Full Mistral OCR response for each source page."),
    ]


class MistralOcrProvider(Protocol):
    async def process(
        self,
        image: EncodedPageImage,
        config: MistralOcrConfig,
        /,
    ) -> MistralOcrProviderResponse: ...


class MistralOcrExecutionError(RuntimeError):
    pass


def build_mistral_ocr_node(_context: object) -> "MistralOcrNode":
    from notarius_plugin_ocr.mistral_sdk import MistralSdkOcrProvider

    return MistralOcrNode(MistralSdkOcrProvider())


@OCR.node(
    operator_id="ocr.mistral.tables",
    version=2,
    title="Mistral OCR 4",
    factory=build_mistral_ocr_node,
)
@final
class MistralOcrNode(
    Node[MistralOcrConfig, MistralOcrInput, MistralOcrOutput]
):
    """Runs Mistral OCR while preserving the full provider response."""

    def __init__(self, provider: MistralOcrProvider) -> None:
        self._provider = provider

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: MistralOcrConfig,
        inputs: MistralOcrInput,
        /,
    ) -> MistralOcrOutput:
        responses: list[MistralOcrResponsePayload] = []
        for sequence_index, image in enumerate(inputs.pages):
            try:
                provider_response = await self._provider.process(image, config)
            except Exception as exc:
                message = (
                    f"Mistral OCR failed for source image {image.filename!r} "
                    f"with model {config.model!r}"
                )
                raise MistralOcrExecutionError(message) from exc

            raw_response = provider_response.raw_response
            if raw_response is None:
                raw_response = cast(
                    JsonObject,
                    provider_response.model_dump(
                        mode="json",
                        by_alias=True,
                    ),
                )
            markdown = "\n".join(
                f"<!-- page {page.index} -->\n{page.markdown.rstrip()}\n"
                for page in provider_response.pages
            )
            responses.append(
                MistralOcrResponsePayload(
                    source_image_artifact_id=image.artifact_id,
                    source_image=image.filename,
                    sequence_index=sequence_index,
                    model=provider_response.model,
                    markdown=markdown,
                    pages=provider_response.pages,
                    usage_info=provider_response.usage_info,
                    raw_response=raw_response,
                )
            )

        return MistralOcrOutput(responses=responses)
