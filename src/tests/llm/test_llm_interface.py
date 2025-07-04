from llm.interface import LLMInterface
from mistralai import Mistral
import logging
import pytest
from logger.setup import setup_logging
setup_logging()
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def sample_base64_image():
    with open("/Users/user/Projects/AI_Osrodek/src/tests/sample_data/0056.jpg", "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

@pytest.fixture(scope="module")
def predictor_interface(cfg):
    """
    Fixture to create and return an instance of LLMInterface.
    This can be used in tests to ensure the client is initialized correctly.
    """
    return LLMInterface(cfg.predictor)

@pytest.fixture(scope="module")
def judge_interface(cfg):
    """
    Fixture to create and return an instance of LLMInterface for the judge.
    This can be used in tests to ensure the judge client is initialized correctly.
    """
    return LLMInterface(cfg.judge)

@pytest.fixture(scope="module")
def predictions():
    return """{
            "page_number": "56",
            "deanery": null,
            "entries": [
                {
                "parish": "Głuchów",
                "dedication": "S. Mathias Ap.",
                "building_material": "mur."
                },
                {
                "parish": "Goszczanów",
                "dedication": "S. Martini EC.",
                "building_material": "lig."
                },
                {
                "parish": "Grodzisko",
                "dedication": "SS. Petr. et Paul. Ap.",
                "building_material": "mur."
                },
                {
                "parish": "Jeżiórsko",
                "dedication": "Exalt. S. Cruc.",
                "building_material": "lig."
                }
            ]
            }
    """

@pytest.fixture(scope="module")    
def ground_truth():
    return """{
            "page_number": "56",
            "deanery": null,
            "entries": [
                {
                "parish": "Głuchów",
                "dedication": "S. Mathias Ap.",
                "building_material": "lig."
                },
                {
                "parish": "Goszczanów",
                "dedication": "S. Martini EC. et S. Stanisl. EM.",
                "building_material": "mur."
                },
                {
                "parish": "Grodzisko",
                "dedication": "SS. Petr. et Paul. Ap.",
                "building_material": "mur."
                },
                {
                "parish": "Jeżiórsko",
                "dedication": "Exalt. S. Cruc.",
                "building_material": "mur."
                }
            ]
            }
    """
    
from PIL import Image
import base64
class TestLLMClient:
    def test_llm_client(self, cfg, sample_base64_image, predictor_interface: LLMInterface):
  

        response = predictor_interface.generate_vision_response(
            base64_image=sample_base64_image,
            system_prompt="system.j2",
            user_prompt="user.j2",
            context={}
        )
        logger.info(f"Response: {response}")
    
        assert response is not None, "Response should not be None"



    def test_llm_judge(self, judge_interface: LLMInterface, predictions, ground_truth):

        response = judge_interface.generate_text_response(
            system_prompt="judge_system.j2",
            user_prompt="judge_input.j2",
            context={
                "prediction": predictions,
                "ground_truth": ground_truth
            }
        )
        
        logger.info(f"Judge Response: {response}")