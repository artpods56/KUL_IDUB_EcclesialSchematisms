"""
Preprocessing steps for the pipeline.

This module provides concrete implementations of preprocessing steps for the pipeline.
"""
from datasets import Dataset

from core.pipeline.steps.base import DatasetProcessingStep
from core.data.filters import filter_schematisms


class SchematismsFilteringStep(DatasetProcessingStep[Dataset, Dataset]):

    def __init__(self, schematisms: list[str]):
        super().__init__()
        self.schematisms = schematisms

    def process_dataset(self, dataset: Dataset) -> Dataset:
        if self.schematisms:
            dataset = dataset.filter(
                filter_schematisms(
                    to_filter=self.schematisms
                ),
                input_columns=["schematism_name"])

            return dataset

        return dataset
