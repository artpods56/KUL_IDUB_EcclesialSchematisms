from typing import Annotated, Protocol, final, override
from uuid import UUID

from PIL.Image import Image
from pydantic import BaseModel, Field

from grafy_core.artifacts import (
    JsonObject,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from grafy_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from grafy_core.operators.images import RASTER_IMAGE

from grafy_plugin_ocr.artifacts import OCR_PAGE_RESULT
from grafy_plugin_ocr.declaration import OCR


class SimpleOcrResult(BaseModel):
    text: str


class OcrEngine[T](Protocol):
    async def recognize(
        self,
        image: Image,
    ) -> T: ...


@final
class FakeOcrEngine(OcrEngine[SimpleOcrResult]):
    @override
    async def recognize(
        self,
        image: Image,
    ) -> SimpleOcrResult:
        return SimpleOcrResult(text=f"fake OCR image {image.width}x{image.height}")


class OcrPagePayload(BaseModel):
    image_artifact_id: UUID
    sequence_index: int
    engine: str
    text: str
    confidence: float | None = None
    language: str | None = None
    provider_metadata: JsonObject = Field(default_factory=dict)


class TesseractOcrInput(NodeInput):
    pages: Annotated[
        list[Image],
        InPort(accepts=RASTER_IMAGE),
        Field(description="Ordered raster images to recognize as pages."),
    ]


class TesseractOcrOutput(NodeOutput):
    results: Annotated[
        list[SimpleOcrResult],
        OutPort(produces=OCR_PAGE_RESULT),
        Field(description="Recognized text for each input page."),
    ]


@OCR.node(
    operator_id="ocr.tesseract.pages",
    version=2,
    title="Tesseract OCR",
    factory=lambda _context: TesseractOcrNode(FakeOcrEngine()),
)
class TesseractOcrNode(Node[NoConfig, TesseractOcrInput, TesseractOcrOutput]):
    """Recognizes plain text from an ordered raster image sequence."""

    def __init__(
        self,
        engine: OcrEngine[SimpleOcrResult],
    ) -> None:
        self._engine = engine

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: TesseractOcrInput,
        /,
    ) -> TesseractOcrOutput:
        results: list[SimpleOcrResult] = []
        for image in inputs.pages:
            result = await self._engine.recognize(image)
            results.append(result)

        return TesseractOcrOutput(results=results)
