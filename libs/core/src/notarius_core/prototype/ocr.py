from typing import Annotated, Any, ClassVar, Protocol, final, override
from uuid import UUID

from PIL.Image import Image
from pydantic import BaseModel, Field

from notarius_core.prototype.artifacts import (
    OCR_PAGE_RESULT,
    SOURCE_PAGE_IMAGE,
    JsonObject,
    NoConfig,
    NodeInput,
    NodeOutput,
)
from notarius_core.prototype.nodes import (
    InPort,
    Node,
    NodeExecutionContext,
    OutPort,
)


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
        InPort(accepts=SOURCE_PAGE_IMAGE),
    ]


class TesseractOcrOutput(NodeOutput):
    results: Annotated[
        list[SimpleOcrResult],
        OutPort(produces=OCR_PAGE_RESULT),
    ]


class TesseractOcrNode(Node[NoConfig, TesseractOcrInput, TesseractOcrOutput]):
    operator_id: ClassVar[str] = "ocr.tesseract.pages"
    operator_version: ClassVar[int] = 1

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


type NodeDefinition = tuple[NodeInput, Node[Any, Any, Any], NodeOutput]


TESSERACT_OCR_NODE: NodeDefinition = (
    TesseractOcrInput,
    TesseractOcrNode,
    TesseractOcrOutput,
)
