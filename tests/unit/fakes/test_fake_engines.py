"""Smoke tests for fake engines.

These tests verify that the fake engines implement the correct interfaces
and provide basic functionality.
"""

import pytest
from PIL import Image

from tests.fakes import FakeOCREngine, FakeLMv3Engine, FakeLLMEngine
from tests.factories import (
    OCRRequestFactory,
    LMv3RequestFactory,
    CompletionRequestFactory,
    OCRResponseFactory,
    SchematismPageFactory,
)
from notarius.infrastructure.ocr.engine_adapter import OCRResponse
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3Response
from notarius.infrastructure.llm.engine_adapter import CompletionResult


class TestFakeOCREngine:
    """Smoke tests for FakeOCREngine."""

    def test_implements_interface(self):
        """Test that FakeOCREngine implements ConfigurableEngine interface."""
        engine = FakeOCREngine()

        # Should have required methods
        assert hasattr(engine, "process")
        assert hasattr(engine, "from_config")
        assert hasattr(engine, "stats")
        assert hasattr(engine, "clear_stats")

    def test_process_returns_ocr_response(self):
        """Test that process() returns OCRResponse."""
        engine = FakeOCREngine(default_text="Test OCR")
        request = OCRRequestFactory.build()

        response = engine.process(request)

        assert isinstance(response, OCRResponse)

    def test_tracks_calls(self):
        """Test that engine tracks all process() calls."""
        engine = FakeOCREngine()
        request1 = OCRRequestFactory.build()
        request2 = OCRRequestFactory.build()

        engine.process(request1)
        engine.process(request2)

        assert len(engine.call_history) == 2
        assert engine.stats["calls"] == 2

    def test_error_simulation(self):
        """Test that engine can simulate errors."""
        engine = FakeOCREngine()
        engine.should_fail = True
        request = OCRRequestFactory.build()

        with pytest.raises(RuntimeError, match="Simulated OCR engine failure"):
            engine.process(request)

        assert engine.stats["errors"] == 1

    def test_custom_response_configuration(self):
        """Test that custom responses can be configured per image."""
        engine = FakeOCREngine()
        image = Image.new("RGB", (100, 100))
        custom_response = OCRResponseFactory.build_with_text("Custom text")

        engine.configure_response(image, custom_response)
        request = OCRRequestFactory.build(input=image)
        response = engine.process(request)

        assert response == custom_response

    def test_reset_clears_state(self):
        """Test that reset() clears all state."""
        engine = FakeOCREngine()
        request = OCRRequestFactory.build()

        engine.process(request)
        engine.should_fail = True

        engine.reset()

        assert len(engine.call_history) == 0
        assert engine.should_fail == False
        assert engine.stats["calls"] == 0


class TestFakeLMv3Engine:
    """Smoke tests for FakeLMv3Engine."""

    def test_implements_interface(self):
        """Test that FakeLMv3Engine implements ConfigurableEngine interface."""
        engine = FakeLMv3Engine()

        assert hasattr(engine, "process")
        assert hasattr(engine, "process_async")
        assert hasattr(engine, "from_config")
        assert hasattr(engine, "stats")

    def test_process_returns_lmv3_response(self):
        """Test that process() returns LMv3Response."""
        engine = FakeLMv3Engine()
        request = LMv3RequestFactory.build()

        response = engine.process(request)

        assert isinstance(response, LMv3Response)
        assert response.output is not None

    @pytest.mark.asyncio
    async def test_process_async_works(self):
        """Test that process_async() works."""
        engine = FakeLMv3Engine()
        request = LMv3RequestFactory.build()

        response = await engine.process_async(request)

        assert isinstance(response, LMv3Response)

    def test_tracks_calls(self):
        """Test that engine tracks all process() calls."""
        engine = FakeLMv3Engine()
        request = LMv3RequestFactory.build()

        engine.process(request)
        engine.process(request)

        assert len(engine.call_history) == 2
        assert engine.stats["calls"] == 2

    def test_custom_default_page(self):
        """Test that custom default page is used."""
        custom_page = SchematismPageFactory.build(entry_count=10)
        engine = FakeLMv3Engine(default_page=custom_page)
        request = LMv3RequestFactory.build()

        response = engine.process(request)

        assert len(response.output.entries) == 10


class TestFakeLLMEngine:
    """Smoke tests for FakeLLMEngine."""

    def test_implements_interface(self):
        """Test that FakeLLMEngine implements ConfigurableEngine interface."""
        engine = FakeLLMEngine()

        assert hasattr(engine, "process")
        assert hasattr(engine, "process_async")
        assert hasattr(engine, "from_config")
        assert hasattr(engine, "stats")

    def test_process_returns_completion_result(self):
        """Test that process() returns CompletionResult."""
        engine = FakeLLMEngine()
        request = CompletionRequestFactory.build()

        result = engine.process(request)

        assert isinstance(result, CompletionResult)

    @pytest.mark.asyncio
    async def test_process_async_works(self):
        """Test that process_async() works."""
        engine = FakeLLMEngine()
        request = CompletionRequestFactory.build()

        result = await engine.process_async(request)

        assert isinstance(result, CompletionResult)

    def test_tracks_calls(self):
        """Test that engine tracks all process() calls."""
        engine = FakeLLMEngine()
        request = CompletionRequestFactory.build()

        engine.process(request)
        engine.process(request)

        assert len(engine.call_history) == 2
        assert engine.stats["calls"] == 2

    def test_response_queue(self):
        """Test that response queue works for sequential responses."""
        from tests.factories import CompletionResultFactory

        engine = FakeLLMEngine()
        request = CompletionRequestFactory.build()

        result1 = CompletionResultFactory.build()
        result2 = CompletionResultFactory.build()

        engine.enqueue_response(result1)
        engine.enqueue_response(result2)

        assert engine.process(request) == result1
        assert engine.process(request) == result2

    def test_error_simulation(self):
        """Test that engine can simulate errors."""
        engine = FakeLLMEngine()
        engine.should_fail = True
        request = CompletionRequestFactory.build()

        with pytest.raises(RuntimeError, match="Simulated LLM engine failure"):
            engine.process(request)

        assert engine.stats["errors"] == 1
