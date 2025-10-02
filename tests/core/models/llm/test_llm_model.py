import io
import json

import pytest
import respx
from httpx import Response

from core.models.llm.model import LLMModel


@pytest.fixture
def mock_openai_api():
    """Fixture that mocks OpenAI API calls for testing"""
    router = respx.mock(assert_all_called=False)
    router.start()

    route = router.post("https://api.openai.com/v1/chat/completions")

    yield route

    router.stop()
    router.reset()


@pytest.fixture(scope="module")
def llm_model(llm_model_config):
    """Fixture that provides an LLMModel instance for testing"""
    return LLMModel(config=llm_model_config, enable_cache=False, test_connection=False)


@pytest.fixture
def llm_model_no_cache(llm_model_config):
    """LLM model with cache disabled"""
    return LLMModel(config=llm_model_config, enable_cache=False, test_connection=False)


class TestLLMModel:

    def test_initialization(self, llm_model):
        """Test proper initialization of LLMModel"""
        assert llm_model
        assert isinstance(llm_model, LLMModel)

    def test_no_input_prediction(self, llm_model):
        """Test that predict raises ValueError because at least one modality (image or text) has to be provided"""
        with pytest.raises(ValueError):
            llm_model.predict()

    def test_text_only_prediction_structured(
        self, llm_model, mock_openai_api, sample_structured_response
    ):
        """Test prediction with text-only input in structured output mode"""
        response_dict = json.loads(sample_structured_response)

        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        response = llm_model.predict(
            text="This is a test text",
        )

        assert response == response_dict

    def test_image_only_prediction_structured(
        self, llm_model, mock_openai_api, sample_structured_response, sample_pil_image
    ):
        """Test prediction with image-only input in structured output mode"""
        response_dict = json.loads(sample_structured_response)

        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        response = llm_model.predict(
            image=sample_pil_image,
        )

        assert response == response_dict

    def test_image_and_text_prediction_structured(
        self, llm_model, mock_openai_api, sample_structured_response, sample_pil_image
    ):
        """Test prediction with both image and text inputs in structured output mode"""
        response_dict = json.loads(sample_structured_response)

        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        response = llm_model.predict(
            image=sample_pil_image,
            text="This is a test text",
        )

        assert response == response_dict

    def test_text_only_prediction_unstructured(
        self, llm_model, mock_openai_api, sample_structured_response, monkeypatch
    ):
        """Test prediction with text-only input in unstructured output mode"""
        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        monkeypatch.setitem(
            llm_model.interface.interface_config, "structured_output", False
        )
        response = llm_model.predict(
            text="This is a test text",
        )

        assert response == {"raw_response": sample_structured_response}

    def test_image_only_prediction_unstructured(
        self,
        llm_model,
        mock_openai_api,
        sample_structured_response,
        sample_pil_image,
        monkeypatch,
    ):
        """Test prediction with image-only input in unstructured output mode"""
        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        monkeypatch.setitem(
            llm_model.interface.interface_config, "structured_output", False
        )
        response = llm_model.predict(
            image=sample_pil_image,
        )

        assert response == {"raw_response": sample_structured_response}

    def test_malformed_json_response(
        self, llm_model, mock_openai_api, malformed_json_response
    ):
        """Test handling of malformed JSON in structured output mode"""
        mock_openai_api.mock(
            return_value=Response(
                200,
                json={"choices": [{"message": {"content": malformed_json_response}}]},
            )
        )

        with pytest.raises(
            ValueError
        ):  # this will always fail because it wont pass if the response is not structured file_format
            response = llm_model.predict(text="test text")

    def test_invalid_image_format(
        self, llm_model, mock_openai_api, sample_structured_response
    ):
        """Test handling of invalid image file_format"""
        # Create an invalid image-like object
        invalid_image = io.BytesIO(b"not an image")

        with pytest.raises(AttributeError):  # Or appropriate exception
            llm_model.predict(image=invalid_image)

    def test_extremely_long_text(
        self, llm_model, mock_openai_api, sample_structured_response
    ):
        """Test handling of very long text input"""
        long_text = "A" * 10000  # Very long text

        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        response = llm_model.predict(text=long_text)
        assert response is not None

    def test_special_characters_in_text(
        self, llm_model, mock_openai_api, sample_structured_response
    ):
        """Test handling of special characters in text"""
        special_text = "Test with émojis 🚀 and spëcial chars: @#$%^&*()"

        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        response = llm_model.predict(text=special_text)
        assert response is not None

    def test_cache_disabled_behavior(
        self, llm_model_config, mock_openai_api, sample_structured_response
    ):
        """Test behavior when cache is disabled"""
        llm_model = LLMModel(
            config=llm_model_config, enable_cache=False, test_connection=False
        )

        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        # Same input should call API twice when cache disabled
        response1 = llm_model.predict(text="test")
        response2 = llm_model.predict(text="test")

        assert mock_openai_api.call_count == 2  # API called twice

    # Integration & Context Tests
    def test_with_hints_parameter(
        self, llm_model, mock_openai_api, sample_structured_response
    ):
        """Test prediction with hints parameter"""
        hints = {"previous_model_output": "some hint"}

        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        response = llm_model.predict(text="test", hints=hints)
        assert response is not None

    def test_with_multiple_kwargs(
        self, llm_model, mock_openai_api, sample_structured_response
    ):
        """Test prediction with multiple additional kwargs"""
        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        response = llm_model.predict(
            text="test",
            hints={"key": "value"},
            context={"additional": "context"},
            metadata={"meta": "data"},
        )

        assert response is not None

    def test_image_and_text_prediction_unstructured(
        self,
        llm_model,
        mock_openai_api,
        sample_structured_response,
        sample_pil_image,
        monkeypatch,
    ):
        """Test prediction with both image and text inputs in unstructured output mode"""
        mock_openai_api.mock(
            return_value=Response(
                200,
                json={
                    "choices": [{"message": {"content": sample_structured_response}}]
                },
            )
        )

        monkeypatch.setitem(
            llm_model.interface.interface_config, "structured_output", False
        )
        response = llm_model.predict(
            image=sample_pil_image,
            text="This is a test text",
        )

        assert response == {"raw_response": sample_structured_response}
