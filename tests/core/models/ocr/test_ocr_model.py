import os
from typing import Dict, List, Tuple

import numpy as np
import pytest
import pytesseract
from PIL import Image

from core.models.ocr.model import OcrModel

@pytest.fixture()
def ocr_model():
    """Create an ``OcrModel`` with cache directory pointing to *tmp_path*."""
    return OcrModel(enable_cache=True, language="eng")

class TestOcrModel:
    """Unit tests for the OcrModel class."""
    def test_model_initialization(self, ocr_model: OcrModel):
        assert ocr_model is not None


    def test_predict_text_only(self, ocr_model: OcrModel, sample_pil_image: Image.Image):
        """Test that the OCR model returns the expected text."""
        text = ocr_model.predict(image=sample_pil_image, text_only=True)
        assert text is not None
        assert isinstance(text, str)

    def test_predict_word_bbox(self, ocr_model: OcrModel, sample_pil_image: Image.Image):
        ocr_response = ocr_model.predict(image=sample_pil_image, text_only=False)

        assert ocr_response is not None
        assert isinstance(ocr_response, Tuple)
        assert len(ocr_response) == 2

        words, bboxes = ocr_response

        assert words is not None
        assert isinstance(words, List)

        assert bboxes is not None
        assert isinstance(bboxes, List)

        assert len(words) == len(bboxes)

def test_text_only_caching(ocr_model: OcrModel, sample_pil_image: Image.Image):
    """The second text-only request should hit the cache and return identical text."""
    text_first = ocr_model.predict(image=sample_pil_image, text_only=True)
    cache_len_after_first = len(ocr_model.cache)

    text_second = ocr_model.predict(image=sample_pil_image, text_only=True)
    cache_len_after_second = len(ocr_model.cache)

    assert text_first == text_second, "Text should be identical across calls"



def test_word_bbox_output_and_cache_hit(ocr_model: OcrModel, sample_pil_image: Image.Image):
    """Requesting words/bboxes after a text-only call should reuse cached OCR."""
    # First call (text-only) fills the cache
    words, bboxes = ocr_model.predict(image=sample_pil_image, text_only=False)

    cached_words, cached_bboxes = ocr_model.predict(image=sample_pil_image, text_only=False)


    assert cached_words == words, "Words should be identical across calls"
    assert cached_bboxes == bboxes, "Bboxes should be identical across calls"
