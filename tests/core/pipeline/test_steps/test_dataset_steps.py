"""Tests for data-wide processing steps."""
from typing import List

import pytest

from core.pipeline.steps.postprocessing import DeaneryFillingStep
from core.schemas.data.pipeline import PipelineData
from core.pipeline.steps.base import DatasetProcessingStep


class CustomDatasetStep(DatasetProcessingStep):
    """Test implementation of DatasetProcessingStep."""
    def process_dataset(self, data: List[PipelineData], **kwargs) -> List[PipelineData]:
        return data


@pytest.fixture
def sample_dataset(sample_pipeline_data) -> List[PipelineData]:
    dataset = [sample_pipeline_data.copy(deep=True) for i in range(5)]

    dataset[1].llm_prediction.entries[0].deanery = "deanery1"
    dataset[3].llm_prediction.entries[0].deanery = "deanery2"
    dataset[4].llm_prediction.entries = []
    return dataset

@pytest.fixture
def multiple_entries_sample_dataset(sample_pipeline_data) -> List[PipelineData]:
    dataset = [sample_pipeline_data.copy(deep=True) for i in range(5)]

    example_entry = dataset[1].llm_prediction.entries[0].copy()
    dataset[1].llm_prediction.entries = [example_entry.copy(deep=True) for i in range(4)]

    dataset[1].llm_prediction.entries[0].deanery = "deanery1"
    dataset[1].llm_prediction.entries[2].deanery = "deanery2"
    return dataset

@pytest.fixture
def deanery_filling_step():
    return DeaneryFillingStep()


class TestDatasetProcessingStepBase:
    def test_description(self):
        """Test the description property."""
        step = CustomDatasetStep()
        assert step.description == "Dataset-level processing CustomDatasetStep"

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            DatasetProcessingStep() # type:ignore


class TestDatasetProcessingStep:
    def test_fixture(self, sample_dataset):
        assert sample_dataset
        assert len(sample_dataset) == 5
        for item in sample_dataset:
            assert isinstance(item, PipelineData)

        assert sample_dataset[1].llm_prediction.entries[0].deanery == "deanery1"

    def test_dataset_wide_processing(self):
        """Test processing that requires full data context."""
        pass

    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        pass

    def test_error_handling(self):
        """Test error handling for invalid data."""
        pass

    def test_data_consistency(self):
        """Test that data-wide operations maintain data consistency."""
        pass

class TestDeaneryFilling:
    def test_step_initialization(self, deanery_filling_step):
        """Test step initialization with various configurations."""
        assert deanery_filling_step
        assert isinstance(deanery_filling_step, DeaneryFillingStep)

    def test_step_execution(self, sample_dataset, deanery_filling_step):

        processed_dataset = deanery_filling_step.process_dataset(sample_dataset)
        assert processed_dataset
        assert len(processed_dataset) == 5
        for item in processed_dataset:
            assert isinstance(item, PipelineData)

    def test_deanery_filling(self, sample_dataset, deanery_filling_step):
        """Test deanery filling step."""
        processed_dataset = deanery_filling_step.process_dataset(sample_dataset)

        assert processed_dataset[0].llm_prediction.entries[0].deanery is None
        assert processed_dataset[1].llm_prediction.entries[0].deanery == "deanery1"
        assert processed_dataset[2].llm_prediction.entries[0].deanery == "deanery1"
        assert processed_dataset[3].llm_prediction.entries[0].deanery == "deanery2"


    def test_multiple_entries_deanery_filling(self, multiple_entries_sample_dataset, deanery_filling_step):

        processed_dataset = deanery_filling_step.process_dataset(multiple_entries_sample_dataset)

        assert processed_dataset[0].llm_prediction.entries[0].deanery is None
        assert processed_dataset[1].llm_prediction.entries[0].deanery == "deanery1"
        assert processed_dataset[1].llm_prediction.entries[1].deanery == "deanery1"
        assert processed_dataset[1].llm_prediction.entries[2].deanery == "deanery2"
        assert processed_dataset[1].llm_prediction.entries[3].deanery == "deanery2"

    def test_empty_dataset(self, deanery_filling_step):
        """Test that empty data doesn't cause errors."""
        result = deanery_filling_step.process_dataset([])
        assert result == []

    def test_none_llm_prediction(self, sample_dataset, deanery_filling_step):
        """Test handling of samples with None llm_prediction."""
        sample_dataset[1].llm_prediction = None
        processed_dataset = deanery_filling_step.process_dataset(sample_dataset)
        # Verify the rest of the data is still processed correctly
        assert processed_dataset[3].llm_prediction.entries[0].deanery == "deanery2"

    def test_empty_entries(self, sample_dataset, deanery_filling_step):
        """Test handling of samples with empty entries list."""
        # Sample with empty entries list already exists in fixture at index 4
        processed_dataset = deanery_filling_step.process_dataset(sample_dataset)
        assert len(processed_dataset[4].llm_prediction.entries) == 0

    def test_deanery_persistence(self, sample_dataset, deanery_filling_step):
        """Test that deanery values persist correctly between batches."""
        # Process first half
        first_half = sample_dataset[:3]
        processed_first = deanery_filling_step.process_dataset(first_half)
        
        # Process second half
        second_half = sample_dataset[3:]
        processed_second = deanery_filling_step.process_dataset(second_half)
        
        # Verify deanery values are consistent
        assert processed_first[1].llm_prediction.entries[0].deanery == "deanery1"
        assert processed_first[2].llm_prediction.entries[0].deanery == "deanery1"
        assert processed_second[0].llm_prediction.entries[0].deanery == "deanery2"

    def test_deanery_overwrite_protection(self, sample_dataset, deanery_filling_step):
        """Test that existing deanery values are not overwritten."""
        original_deanery = "original_deanery"
        sample_dataset[2].llm_prediction.entries[0].deanery = original_deanery
        
        processed_dataset = deanery_filling_step.process_dataset(sample_dataset)
        assert processed_dataset[2].llm_prediction.entries[0].deanery == original_deanery