"""Tests for preprocessing steps like language detection."""
from core.pipeline.steps.prediction import LanguageDetectionStep
from core.schemas.data.schematism import SchematismPage
import pytest
from PIL import Image
import numpy as np
from unittest.mock import patch


from core.schemas.data.pipeline import PipelineData
from core.models.ocr.model import OcrModel

from tests.conftest import sample_page_data


@pytest.fixture
def sample_text_pl():
    return "To jest przykładowy polski tekst do wykrycia języka."

@pytest.fixture
def sample_text_de():
    return "Dies ist ein Beispieltext auf Deutsch zur Spracherkennung."

@pytest.fixture
def dummy_image():
    # Create a small dummy image
    return Image.fromarray(np.zeros((100, 100), dtype=np.uint8))

@pytest.fixture
def pipeline_data_with_text(sample_text_pl, dummy_image, sample_page_data: SchematismPage):
    return PipelineData(
        image=dummy_image,
        text=sample_text_pl,
        ground_truth=sample_page_data,  # Mock ground truth as it's required but not used in these tests
        language=None,
        language_confidence=None,
        lmv3_prediction=None,
        llm_prediction=None
    )

@pytest.fixture
def pipeline_data_without_text(dummy_image, sample_page_data: SchematismPage):
    return PipelineData(
        image=dummy_image,
        text=None,
        ground_truth=sample_page_data,
        language=None,
        language_confidence=None,
        lmv3_prediction=None,
        llm_prediction=None
    )

class TestLanguageDetectionStep:
    def test_successful_language_detection(self, pipeline_data_with_text):
        """Test successful language detection for a single sample."""
        step = LanguageDetectionStep(languages=["POLISH", "GERMAN"])
        result = step.process(pipeline_data_with_text)
        
        assert result.language == "POLISH"
        assert isinstance(result.language_confidence, float)
        assert 0 <= result.language_confidence <= 1

    def test_multiple_languages_detection(self, sample_page_data: SchematismPage, pipeline_data_with_text, sample_text_de, dummy_image):
        """Test language detection across multiple samples."""
        step = LanguageDetectionStep(languages=["POLISH", "GERMAN"])
        
        # Create a German sample
        german_data = PipelineData(
            image=dummy_image,
            text=sample_text_de,
            ground_truth=sample_page_data,
            language=None,
            language_confidence=None,
            lmv3_prediction=None,
            llm_prediction=None
        )
        
        # Process both samples
        result_pl = step.process(pipeline_data_with_text)
        result_de = step.process(german_data)
        
        assert result_pl.language == "POLISH"
        assert result_de.language == "GERMAN"

    @patch.object(OcrModel, 'predict')
    def test_fallback_to_ocr(self, mock_predict, pipeline_data_without_text):
        """Test that step falls back to OCR when no text is provided."""
        mock_predict.return_value = "Sample OCR text"
        
        step = LanguageDetectionStep(languages=["POLISH", "GERMAN"])
        result = step.process(pipeline_data_without_text)
        
        mock_predict.assert_called_once()
        assert result.text == "Sample OCR text"
        assert result.language is not None

    def test_batch_processing(self, pipeline_data_with_text, sample_text_de, dummy_image, sample_page_data):
        """Test batch processing of multiple samples."""
        step = LanguageDetectionStep(languages=["POLISH", "GERMAN"])
        
        # Create a batch with both Polish and German samples
        german_data = PipelineData(
            image=dummy_image,
            text=sample_text_de,
            ground_truth=sample_page_data,
            language=None,
            language_confidence=None,
            lmv3_prediction=None,
            llm_prediction=None
        )
        batch = [pipeline_data_with_text, german_data]
        
        results = step.batch_process(batch)
        assert len(results) == 2
        assert results[0].language == "POLISH"
        assert results[1].language == "GERMAN"

    def test_error_handling_invalid_input(self, dummy_image, sample_page_data: SchematismPage):
        """Test handling of invalid input data."""
        step = LanguageDetectionStep(languages=["POLISH", "GERMAN"])
        
        # Test with no text and no image
        invalid_data = PipelineData(
            image=None,
            text=None,
            ground_truth=sample_page_data,
            language=None,
            language_confidence=None,
            lmv3_prediction=None,
            llm_prediction=None
        )
        
        with pytest.raises(ValueError, match="No image provided for OCR processing"):
            step.process(invalid_data)

    def test_confidence_score_range(self, pipeline_data_with_text):
        """Test that confidence scores are within expected range."""
        step = LanguageDetectionStep(languages=["POLISH", "GERMAN"])
        result = step.process(pipeline_data_with_text)
        
        assert result.language_confidence is not None
        assert 0 <= result.language_confidence <= 1

    def test_invalid_language_config(self):
        """Test handling of invalid language configuration."""
        step = LanguageDetectionStep(languages=["INVALID_LANG"])
        # Should fall back to all languages when invalid language is provided
        assert step.detector is not None

    def test_text_preprocessing(self, pipeline_data_with_text):
        """Test text preprocessing functionality."""
        step = LanguageDetectionStep(languages=["POLISH"])
        
        # Add some numbers and extra whitespace to the text
        pipeline_data_with_text.text = "123 Test   text\n  with   numbers 456"
        result = step.process(pipeline_data_with_text)
        
        # The processed text should have normalized whitespace and no numbers
        processed_text = step._preprocess_text(pipeline_data_with_text.text)
        assert "123" not in processed_text
        assert "456" not in processed_text
        assert "\n" not in processed_text
        assert "  " not in processed_text
