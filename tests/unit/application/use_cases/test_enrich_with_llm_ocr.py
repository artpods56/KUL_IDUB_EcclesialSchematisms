"""Tests for EnrichDatasetWithLLMOCR use case."""

import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch
from PIL import Image

from notarius.application.use_cases.inference.enrich_dataset_with_ocr_using_llm import (
    EnrichDatasetWithLLMOCR,
    EnrichWithLLMOCRRequest,
    EnrichWithLLMOCRResponse,
)
from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.infrastructure.llm.engine_adapter import (
    LLMEngine,
    CompletionResult,
)
from notarius.infrastructure.llm.conversation import Conversation
from notarius.domain.entities.completions import BaseProviderResponse
from notarius.domain.entities.messages import ChatMessage, TextContent
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem, BaseMetaData


# Mock provider response for testing
@dataclass(frozen=True)
class MockLLMTextResponse(BaseProviderResponse[None]):
    """Mock LLM provider response for text-only output."""

    def to_string(self) -> str:
        return self.text_response or ""


# Test fixtures


@pytest.fixture
def mock_image() -> Image.Image:
    """Create a mock PIL image."""
    return Image.new("RGB", (100, 100), color="white")


@pytest.fixture
def mock_llm_engine() -> MagicMock:
    """Create a mock LLM engine."""
    engine = MagicMock(spec=LLMEngine)

    mock_response = MockLLMTextResponse(
        structured_response=None,
        text_response="Extracted OCR text from LLM",
    )
    mock_conversation = Conversation.from_messages(
        [ChatMessage(role="assistant", content=[TextContent(text="response")])]
    )

    engine.process.return_value = CompletionResult(
        output=mock_response,
        conversation=mock_conversation,
    )
    return engine


@pytest.fixture
def mock_image_storage(mock_image: Image.Image) -> MagicMock:
    """Create a mock image storage resource."""
    storage = MagicMock()
    storage.load_image.return_value = mock_image
    return storage


@pytest.fixture
def mock_prompt_renderer() -> MagicMock:
    """Create a mock Jinja2 prompt renderer."""
    renderer = MagicMock()
    renderer.render_prompt.return_value = "Rendered prompt text"
    return renderer


@pytest.fixture
def sample_metadata() -> BaseMetaData:
    """Create sample metadata."""
    return BaseMetaData(
        sample_id=1,
        schematism_name="test_schematism",
        filename="test_file.jpg",
    )


@pytest.fixture
def sample_dataset(sample_metadata: BaseMetaData) -> BaseDataset[BaseDataItem]:
    """Create a sample dataset with items."""
    items = [
        BaseDataItem(
            image_path="/path/to/image1.jpg",
            text=None,
            metadata=sample_metadata,
        ),
        BaseDataItem(
            image_path="/path/to/image2.jpg",
            text=None,
            metadata=BaseMetaData(
                sample_id=2,
                schematism_name="test_schematism",
                filename="test_file2.jpg",
            ),
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


@pytest.fixture
def empty_dataset() -> BaseDataset[BaseDataItem]:
    """Create an empty dataset."""
    return BaseDataset[BaseDataItem](items=[])


@pytest.fixture
def dataset_with_missing_paths(
    sample_metadata: BaseMetaData,
) -> BaseDataset[BaseDataItem]:
    """Create a dataset with items missing image paths."""
    items = [
        BaseDataItem(image_path=None, text=None, metadata=sample_metadata),
        BaseDataItem(
            image_path="/path/to/image.jpg", text=None, metadata=sample_metadata
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


class TestEnrichWithLLMOCRRequest:
    """Test suite for EnrichWithLLMOCRRequest dataclass."""

    def test_request_with_defaults(
        self, sample_dataset: BaseDataset[BaseDataItem]
    ) -> None:
        """Test request creation with default prompts."""
        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)

        assert request.dataset is sample_dataset
        assert request.system_prompt == "tasks/ocr/system.j2"
        assert request.user_prompt == "tasks/ocr/user.j2"

    def test_request_with_custom_prompts(
        self, sample_dataset: BaseDataset[BaseDataItem]
    ) -> None:
        """Test request creation with custom prompts."""
        request = EnrichWithLLMOCRRequest(
            dataset=sample_dataset,
            system_prompt="custom_system.j2",
            user_prompt="custom_user.j2",
        )

        assert request.system_prompt == "custom_system.j2"
        assert request.user_prompt == "custom_user.j2"


class TestEnrichWithLLMOCRResponse:
    """Test suite for EnrichWithLLMOCRResponse dataclass."""

    def test_response_creation(self, sample_dataset: BaseDataset[BaseDataItem]) -> None:
        """Test response creation with all fields."""
        response = EnrichWithLLMOCRResponse(
            dataset=sample_dataset,
            llm_executions=5,
            cache_hits=3,
            success_rate=0.8,
        )

        assert response.dataset is sample_dataset
        assert response.llm_executions == 5
        assert response.cache_hits == 3
        assert response.success_rate == 0.8


class TestEnrichDatasetWithLLMOCR:
    """Test suite for EnrichDatasetWithLLMOCR use case."""

    def test_init_with_cache_disabled(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test initialization with caching disabled."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            enable_cache=False,
        )

        assert use_case.llm_engine is mock_llm_engine
        assert use_case.image_storage is mock_image_storage

    @patch(
        "notarius.application.use_cases.inference.add_llm_ocr_to_dataset.create_llm_cache_backend"
    )
    def test_init_with_cache_enabled(
        self,
        mock_create_cache: MagicMock,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test initialization with caching enabled."""
        mock_backend = MagicMock()
        mock_keygen = MagicMock()
        mock_create_cache.return_value = (mock_backend, mock_keygen)

        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            enable_cache=True,
        )

        assert isinstance(use_case.llm_engine, CachedEngine)
        mock_create_cache.assert_called_once_with("test-model")

    def test_init_with_custom_prompt_renderer(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
    ) -> None:
        """Test initialization with custom prompt renderer."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        assert use_case.prompt_renderer is mock_prompt_renderer

    async def test_execute_processes_all_items(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute processes all dataset items."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert mock_llm_engine.process.call_count == 2
        assert mock_image_storage.load_image.call_count == 2

    async def test_execute_extracts_text_from_llm_response(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute extracts text from LLM response."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        # All items should have the extracted text
        for item in response.dataset.items:
            assert item.text == "Extracted OCR text from LLM"

    async def test_execute_preserves_metadata(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute preserves item metadata."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert response.dataset.items[0].metadata.sample_id == 1
        assert response.dataset.items[1].metadata.sample_id == 2

    async def test_execute_preserves_image_paths(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute preserves image paths."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert response.dataset.items[0].image_path == "/path/to/image1.jpg"
        assert response.dataset.items[1].image_path == "/path/to/image2.jpg"

    async def test_execute_skips_items_without_image_path(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        dataset_with_missing_paths: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that items without image paths are kept but not processed."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=dataset_with_missing_paths)
        response = await use_case.execute(request)

        # Both items should be in the output
        assert len(response.dataset.items) == 2
        # Only one was processed by the LLM
        assert mock_llm_engine.process.call_count == 1

    async def test_execute_with_empty_dataset(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        empty_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execution with empty dataset."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=empty_dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 0
        assert response.llm_executions == 0
        assert response.cache_hits == 0
        assert response.success_rate == 0.0
        mock_llm_engine.process.assert_not_called()

    async def test_execute_returns_correct_statistics_without_cache(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that statistics are correct when cache is disabled."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        assert response.llm_executions == 2
        assert response.cache_hits == 0
        assert response.success_rate == 1.0

    async def test_execute_uses_no_structured_output(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that execute requests no structured output (text-only)."""
        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        await use_case.execute(request)

        # Verify CompletionRequest was created with structured_output=None
        call_args = mock_llm_engine.process.call_args_list
        for call in call_args:
            completion_request = call[0][0]
            assert completion_request.structured_output is None

    async def test_execute_converts_image_to_rgb(
        self,
        mock_llm_engine: MagicMock,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that images are converted to RGB mode."""
        grayscale_image = Image.new("L", (100, 100), color=128)
        rgb_image = Image.new("RGB", (100, 100), color="white")
        grayscale_image.convert = MagicMock(return_value=rgb_image)
        mock_image_storage.load_image.return_value = grayscale_image

        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_llm_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        await use_case.execute(request)

        grayscale_image.convert.assert_called_with("RGB")

    async def test_execute_handles_llm_error_gracefully(
        self,
        mock_image_storage: MagicMock,
        mock_prompt_renderer: MagicMock,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that LLM errors are handled gracefully."""
        mock_engine = MagicMock(spec=LLMEngine)
        mock_engine.process.side_effect = Exception("LLM processing failed")

        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_prompt_renderer,
            enable_cache=False,
        )

        request = EnrichWithLLMOCRRequest(dataset=sample_dataset)
        response = await use_case.execute(request)

        # Items should be preserved (original items kept on failure)
        assert len(response.dataset.items) == 2
        # Success rate should be 0 due to failures
        assert response.success_rate == 0.0


class TestEnrichDatasetWithLLMOCRIntegration:
    """Integration-style tests for EnrichDatasetWithLLMOCR."""

    async def test_full_workflow_produces_valid_output(
        self,
        mock_image_storage: MagicMock,
    ) -> None:
        """Test complete workflow produces valid BaseDataItems with OCR text."""
        mock_engine = MagicMock(spec=LLMEngine)
        mock_renderer = MagicMock()
        mock_renderer.render_prompt.return_value = "Rendered prompt"

        # Return different text for each page
        mock_response1 = MockLLMTextResponse(
            structured_response=None,
            text_response="# Page 1 Header\n\nContent from page 1",
        )
        mock_response2 = MockLLMTextResponse(
            structured_response=None,
            text_response="# Page 2 Header\n\nContent from page 2",
        )

        mock_conversation = Conversation.from_messages(
            [ChatMessage(role="assistant", content=[TextContent(text="resp")])]
        )

        mock_engine.process.side_effect = [
            CompletionResult(output=mock_response1, conversation=mock_conversation),
            CompletionResult(output=mock_response2, conversation=mock_conversation),
        ]

        use_case = EnrichDatasetWithLLMOCR(
            llm_engine=mock_engine,
            image_storage=mock_image_storage,
            model_name="test-model",
            prompt_renderer=mock_renderer,
            enable_cache=False,
        )

        dataset = BaseDataset[BaseDataItem](
            items=[
                BaseDataItem(
                    image_path="/path/page1.jpg",
                    text=None,
                    metadata=BaseMetaData(
                        sample_id=1, schematism_name="test", filename="p1.jpg"
                    ),
                ),
                BaseDataItem(
                    image_path="/path/page2.jpg",
                    text=None,
                    metadata=BaseMetaData(
                        sample_id=2, schematism_name="test", filename="p2.jpg"
                    ),
                ),
            ]
        )

        request = EnrichWithLLMOCRRequest(dataset=dataset)
        response = await use_case.execute(request)

        assert len(response.dataset.items) == 2
        assert (
            response.dataset.items[0].text == "# Page 1 Header\n\nContent from page 1"
        )
        assert (
            response.dataset.items[1].text == "# Page 2 Header\n\nContent from page 2"
        )
        assert response.llm_executions == 2
        assert response.cache_hits == 0
        assert response.success_rate == 1.0
