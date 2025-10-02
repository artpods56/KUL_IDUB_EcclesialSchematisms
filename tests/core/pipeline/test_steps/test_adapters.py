import json

from core.pipeline.steps.transformation import HuggingFaceTransformationStep
import pytest
from omegaconf import DictConfig, OmegaConf
from core.schemas.data.pipeline import PipelineData
from core.schemas.data.schematism import SchematismPage

class TestHFIngestionAdapter:
    """Tests for the HFIngestionAdapter class."""

    def test_successful_initialization(self, dataset_config: DictConfig):
        """
        Tests that the adapter initializes correctly with a valid config fixture.
        """
        try:
            adapter = HuggingFaceTransformationStep(dataset_config=dataset_config)
            assert adapter is not None
            # Check that the source column names were loaded correctly from the config
            assert adapter.image_src_col == "image"
            assert adapter.ground_truth_src_col == "results"
        except ValueError:
            pytest.fail("HFIngestionAdapter raised ValueError unexpectedly during initialization.")

    def test_failed_initialization_missing_mapping(self):
        """
        Tests that initialization fails if the config is missing a required column mapping.
        """
        # Create a config that is missing the 'ground_truth_column' mapping
        bad_config = OmegaConf.create({
            "column_map": {
                "image_column": "image"
                # The 'ground_truth_column' is missing
            }
        })
        with pytest.raises(ValueError, match="Config's column_map must specify 'image_column' and 'ground_truth_column'."):
            HuggingFaceTransformationStep(dataset_config=bad_config)

    def test_successful_processing(self, dataset_config: DictConfig, sample_pil_image, sample_structured_response):
        """
        Tests the successful processing of a single, valid data sample.
        The test uses real fixtures to ensure the adapter works with the project's configuration.
        """
        adapter = HuggingFaceTransformationStep(dataset_config=dataset_config)
        
        # The 'schematism_dataset_config' maps 'ground_truth_column' to 'results'.
        # So, the raw data sample must have an 'image' key and a 'results' key.
        sample_data = {
            "image": sample_pil_image,
            "results": sample_structured_response
        }
        
        pipeline_data = adapter.process(sample_data)
        
        assert isinstance(pipeline_data, PipelineData)
        assert pipeline_data.image == sample_pil_image

        assert isinstance(pipeline_data.ground_truth, SchematismPage)
        assert isinstance(sample_structured_response, str)
        assert pipeline_data.ground_truth.model_dump() == json.loads(sample_structured_response)
        # assert pipeline_data.ground_truth == sample_structured_response
        # Verify the parsed ground_truth data structure

    def test_failed_processing_missing_data(self, dataset_config: DictConfig, sample_pil_image):
        """
        Tests that processing fails if the data sample is missing a required field
        (in this case, the ground truth).
        """
        adapter = HuggingFaceTransformationStep(dataset_config=dataset_config)
        
        # This sample is missing the 'results' key, which is mapped to ground_truth.
        bad_sample_data = {
            "image": sample_pil_image
        }
        
        with pytest.raises(ValueError, match="Missing required data. Looking for 'image' and 'results' in the data sample."):
            adapter.process(bad_sample_data)

    def test_batch_processing(self, dataset_config: DictConfig, sample_pil_image, sample_structured_response):
        """
        Tests the successful batch processing of multiple valid data samples.
        """
        adapter = HuggingFaceTransformationStep(dataset_config=dataset_config)
        
        sample_data_1 = {"image": sample_pil_image, "results": sample_structured_response}
        sample_data_2 = {"image": sample_pil_image, "results": sample_structured_response}
        
        batch_data = [sample_data_1, sample_data_2]
        
        processed_batch = adapter.batch_process(batch_data)
        
        assert isinstance(processed_batch, list)
        assert len(processed_batch) == 2
        for pipeline_data in processed_batch:
            assert isinstance(pipeline_data, PipelineData)
            assert pipeline_data.image == sample_pil_image
            # assert pipeline_data.ground_truth == sample_structured_response
