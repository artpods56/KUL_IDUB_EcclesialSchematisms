import logging

from llm.prompt_manager import prompt_manager
from logger.setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class TestPromptManager:

    def test_object_creation(self):
        assert prompt_manager is not None, "PromptManager instance should be created"

    def test_render_prompt(self):

        rendered = prompt_manager.render_prompt("system.j2")
        assert rendered is not None, "Rendered prompt should not be None"

    def test_dataset_sample_fixture(self, dataset_sample):
        """
        Test to ensure the dataset sample fixture is working correctly.
        """
        logger.info("Testing dataset sample fixture.")
        logger.info(f"Dataset sample type: {type(dataset_sample)}")
        logger.info(dataset_sample)

    def test_local_dataset_sample_fixture(self, local_dataset_sample):
        """
        Test to ensure the dataset sample fixture is working correctly.
        """
        logger.info("Testing dataset sample fixture.")
        logger.info(f"Dataset sample type: {type(local_dataset_sample)}")
        logger.info(local_dataset_sample)