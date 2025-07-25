import PIL
import pytest
import json
import respx
import io

from PIL.Image import Image

from core.models.lmv3.model import  LMv3Model


@pytest.fixture(scope="module")
def lmv3_model(lmv3_model_config):
    """Fixture that provides an LLMModel instance for testing"""
    return LMv3Model(
        config=lmv3_model_config,
    )


@pytest.fixture
def lmv3_model_no_cache(llm_model_config):
    """LLM model with cache disabled"""
    return LMv3Model(config=llm_model_config, enable_cache=False)


class TestLLMModel:

    def test_initialization(self, lmv3_model):
        """Test proper initialization of LLMModel"""
        assert lmv3_model
        assert isinstance(lmv3_model, LMv3Model)

    def test_no_input_prediction(self, lmv3_model):
        """Test that predict raises ValueError because at least one modality (image or text) has to be provided"""
        with pytest.raises(TypeError):
            lmv3_model.predict()

    def test_predict_text_only(self, lmv3_model: LMv3Model, sample_pil_image: Image):
        """Test that the OCR model returns the expected text."""
        text = lmv3_model.predict(pil_image=sample_pil_image)
        assert text is not None
        assert isinstance(text, dict)

