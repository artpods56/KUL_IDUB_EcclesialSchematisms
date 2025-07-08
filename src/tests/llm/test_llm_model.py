import pytest
import respx
from httpx import Response

from core.models.llm.model import LLMModel


@pytest.fixture(autouse=True)
def mock_openai_api():
    # Start mocking
    router = respx.mock(assert_all_called=False, assert_all_mocked=False)
    router.start()

    route = router.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            status_code=200,
            json={
                "choices": [
                    {"message": {"content": "mocked response content"}}
                ]
            }
        )
    )

    yield route

    # Stop mocking after test
    router.stop()
    router.reset()


@pytest.fixture(scope="module")
def llm_model(llm_model_config):
    return LLMModel(
        config=llm_model_config,
        enable_cache=False,
        test_connection=False
    )


class TestLLMModel:

    def test_initialization(self, llm_model):
        assert llm_model
        assert isinstance(llm_model, LLMModel)

    def test_text_only_prediction(self, llm_model):
        llm_model.interface.interface_config["structured_output"] = False
        response = llm_model.predict(
            text="This is a test text",
            hints={"sample": "sample"}
        )

        assert response == {'raw_response': 'mocked response content'}
