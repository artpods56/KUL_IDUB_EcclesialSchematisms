"""Fake engine implementations for testing.

This module provides realistic fake implementations of engines that replace mocks.
Fakes implement the actual ConfigurableEngine interface and provide call tracking,
error simulation, and configurable responses WITHOUT external dependencies.
"""

from typing import Self, Any
from dataclasses import dataclass
from PIL import Image
from pydantic import BaseModel

from notarius.application.ports.outbound.engine import (
    ConfigurableEngine,
    track_stats,
    track_stats_async,
)
from notarius.infrastructure.ocr.engine_adapter import OCRRequest, OCRResponse
from notarius.infrastructure.ocr.types import SimpleOCRResult, StructuredOCRResult
from notarius.infrastructure.ml_models.lmv3.engine_adapter import (
    LMv3Request,
    LMv3Response,
)
from notarius.infrastructure.llm.engine_adapter import (
    CompletionRequest,
    CompletionResult,
)
from notarius.domain.entities.schematism import SchematismPage
from notarius.domain.entities.completions import BaseProviderResponse
from notarius.schemas.configs import (
    PytesseractOCRConfig,
    BaseLMv3ModelConfig,
    LLMEngineConfig,
)

from tests.factories.responses import (
    OCRResponseFactory,
    LMv3ResponseFactory,
    BaseProviderResponseFactory,
    CompletionResultFactory,
)
from tests.factories.entities import SchematismPageFactory


class FakeOCREngine(ConfigurableEngine[PytesseractOCRConfig, OCRRequest, OCRResponse]):
    """Fake OCR engine for testing without PyTesseract dependency.

    This fake engine:
    - Implements the real OCREngine interface
    - Tracks all process() calls
    - Supports error simulation
    - Allows custom response configuration
    - Tracks stats automatically via @track_stats

    Example:
        # Basic usage
        engine = FakeOCREngine(default_text="Sample OCR")
        response = engine.process(OCRRequest(input=image, mode="text"))
        assert response.output.text == "Sample OCR"

        # Error simulation
        engine.should_fail = True
        with pytest.raises(RuntimeError):
            engine.process(request)

        # Custom responses
        engine.configure_response(image, custom_response)
        response = engine.process(OCRRequest(input=image, mode="text"))
        assert response == custom_response

        # Call tracking
        assert len(engine.call_history) == 2
        assert engine.stats["calls"] == 2
    """

    def __init__(
        self,
        default_text: str = "Fake OCR output",
        config: PytesseractOCRConfig | None = None,
    ):
        """Initialize fake OCR engine.

        Args:
            default_text: Default text to return for simple OCR
            config: Optional OCR configuration (not used, but matches interface)
        """
        self._init_stats()
        self.config = config
        self.default_text = default_text
        self.call_history: list[OCRRequest] = []
        self.should_fail = False
        self.custom_responses: dict[int, OCRResponse] = {}

    @classmethod
    def from_config(cls, config: PytesseractOCRConfig) -> Self:
        """Create FakeOCREngine from config.

        Args:
            config: OCR configuration

        Returns:
            New FakeOCREngine instance
        """
        return cls(config=config)

    @track_stats
    def process(self, request: OCRRequest) -> OCRResponse:
        """Process OCR request with fake response.

        Args:
            request: OCR request with image and mode

        Returns:
            OCR response with fake text or structured output

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_history.append(request)

        if self.should_fail:
            raise RuntimeError("Simulated OCR engine failure")

        # Check for custom response
        image_id = id(request.input)
        if image_id in self.custom_responses:
            return self.custom_responses[image_id]

        # Generate default response based on mode
        return OCRResponseFactory.build(mode=request.mode, text=self.default_text)

    def configure_response(self, image: Image.Image, response: OCRResponse) -> None:
        """Configure a custom response for a specific image.

        Args:
            image: The PIL Image to configure
            response: The OCR response to return for this image
        """
        self.custom_responses[id(image)] = response

    def reset(self) -> None:
        """Reset call history and state."""
        self.call_history.clear()
        self.should_fail = False
        self.custom_responses.clear()
        self.clear_stats()


class FakeLMv3Engine(
    ConfigurableEngine[BaseLMv3ModelConfig, LMv3Request, LMv3Response]
):
    """Fake LayoutLMv3 engine for testing without transformers dependency.

    This fake engine:
    - Implements the real LMv3Engine interface
    - Tracks all process() and process_async() calls
    - Supports both sync and async processing
    - Allows custom page responses
    - Tracks stats automatically

    Example:
        # Basic usage
        engine = FakeLMv3Engine()
        response = engine.process(LMv3Request(input=image))
        assert len(response.output.entries) > 0

        # Custom page
        custom_page = SchematismPageFactory.build(entry_count=10)
        engine = FakeLMv3Engine(default_page=custom_page)
        response = engine.process(LMv3Request(input=image))
        assert len(response.output.entries) == 10

        # Async support
        response = await engine.process_async(request)

        # Call tracking
        assert len(engine.call_history) == 2
    """

    def __init__(
        self,
        default_page: SchematismPage | None = None,
        config: BaseLMv3ModelConfig | None = None,
    ):
        """Initialize fake LMv3 engine.

        Args:
            default_page: Default page to return (auto-generated if not provided)
            config: Optional LMv3 configuration (not used, but matches interface)
        """
        self._init_stats()
        self.config = config
        self.default_page = default_page or SchematismPageFactory.build()
        self.call_history: list[LMv3Request] = []
        self.should_fail = False

    @classmethod
    def from_config(cls, config: BaseLMv3ModelConfig) -> Self:
        """Create FakeLMv3Engine from config.

        Args:
            config: LMv3 configuration

        Returns:
            New FakeLMv3Engine instance
        """
        return cls(config=config)

    @track_stats
    def process(self, request: LMv3Request) -> LMv3Response:
        """Process LMv3 request with fake response.

        Args:
            request: LMv3 request with image

        Returns:
            LMv3 response with fake predictions

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_history.append(request)

        if self.should_fail:
            raise RuntimeError("Simulated LMv3 engine failure")

        return LMv3ResponseFactory.build(output=self.default_page)

    @track_stats_async
    async def process_async(self, request: LMv3Request) -> LMv3Response:
        """Process LMv3 request asynchronously with fake response.

        Args:
            request: LMv3 request with image

        Returns:
            LMv3 response with fake predictions

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_history.append(request)

        if self.should_fail:
            raise RuntimeError("Simulated LMv3 engine failure")

        return LMv3ResponseFactory.build(output=self.default_page)

    def reset(self) -> None:
        """Reset call history and state."""
        self.call_history.clear()
        self.should_fail = False
        self.clear_stats()


@dataclass(frozen=True)
class FakeProviderResponse[T: BaseModel](BaseProviderResponse[T]):
    """Concrete implementation of BaseProviderResponse for FakeLLMEngine."""

    pass


class FakeLLMEngine(
    ConfigurableEngine[LLMEngineConfig, CompletionRequest, CompletionResult]
):
    """Fake LLM engine for testing without API calls.

    This fake engine:
    - Implements the real LLMEngine interface
    - Preserves generic type parameters
    - Supports both sync and async processing
    - Allows response queue for sequential responses
    - Supports structured output
    - Tracks conversation history

    Example:
        # Basic usage
        engine = FakeLLMEngine()
        request = CompletionRequestFactory.build()
        result = engine.process(request)

        # Structured output
        from pydantic import BaseModel
        class MySchema(BaseModel):
            field: str

        request = CompletionRequestFactory.build(structured_output=MySchema)
        result = engine.process(request)
        # result.output.structured_response will be a MySchema instance

        # Queue responses
        engine.enqueue_response(custom_result1)
        engine.enqueue_response(custom_result2)
        result1 = engine.process(request)  # Gets custom_result1
        result2 = engine.process(request)  # Gets custom_result2

        # Call tracking
        assert len(engine.call_history) == 2
    """

    def __init__(self, config: LLMEngineConfig | None = None):
        """Initialize fake LLM engine.

        Args:
            config: Optional LLM configuration (not used, but matches interface)
        """
        self._init_stats()
        self.config = config
        self.call_history: list[CompletionRequest] = []
        self.response_queue: list[CompletionResult] = []
        self.should_fail = False

    @classmethod
    def from_config(cls, config: LLMEngineConfig) -> Self:
        """Create FakeLLMEngine from config.

        Args:
            config: LLM configuration

        Returns:
            New FakeLLMEngine instance
        """
        return cls(config=config)

    def enqueue_response(self, response: CompletionResult) -> None:
        """Enqueue a response to be returned in sequence.

        Args:
            response: The completion result to enqueue
        """
        self.response_queue.append(response)

    @track_stats
    def process[T: BaseModel](
        self, request: CompletionRequest[T]
    ) -> CompletionResult[T]:
        """Process LLM completion request with fake response.

        Args:
            request: Completion request with conversation and optional structured output

        Returns:
            Completion result with fake LLM response

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_history.append(request)

        if self.should_fail:
            raise RuntimeError("Simulated LLM engine failure")

        # Return queued response if available
        if self.response_queue:
            return self.response_queue.pop(0)

        # Generate default response
        if request.structured_output:
            # Create a fake instance of the structured output type
            # This is a simple implementation - in real tests you'd configure this
            return CompletionResultFactory.build(
                conversation=request.input, structured_output_expected=True
            )
        else:
            return CompletionResultFactory.build(
                conversation=request.input, structured_output_expected=False
            )

    @track_stats_async
    async def process_async[T: BaseModel](
        self, request: CompletionRequest[T]
    ) -> CompletionResult[T]:
        """Process LLM completion request asynchronously with fake response.

        Args:
            request: Completion request with conversation and optional structured output

        Returns:
            Completion result with fake LLM response

        Raises:
            RuntimeError: If should_fail is True
        """
        self.call_history.append(request)

        if self.should_fail:
            raise RuntimeError("Simulated LLM engine failure")

        # Return queued response if available
        if self.response_queue:
            return self.response_queue.pop(0)

        # Generate default response
        if request.structured_output:
            return CompletionResultFactory.build(
                conversation=request.input, structured_output_expected=True
            )
        else:
            return CompletionResultFactory.build(
                conversation=request.input, structured_output_expected=False
            )

    def reset(self) -> None:
        """Reset call history, queue, and state."""
        self.call_history.clear()
        self.response_queue.clear()
        self.should_fail = False
        self.clear_stats()
