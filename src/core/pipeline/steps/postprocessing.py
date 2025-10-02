"""Post-processing steps for pipeline data refinement and completion.

This module contains data-level processing steps that perform cross-sample
analysis and data completion tasks that require access to the entire data context.
"""

from typing import Optional, Any

from core.data.translation_parser import Parser
from core.pipeline.steps.base import DatasetProcessingStep
from core.schemas.data.pipeline import PipelineData
from core.schemas.data.schematism import SchematismPage



class DeaneryFillingStep(DatasetProcessingStep[list[PipelineData], list[PipelineData]]):

    def __init__(self, sources: list[str]):
        super().__init__()

        self.sources = sources

    def process_dataset(self, dataset: list[PipelineData]) -> list[PipelineData]:
        for source in self.sources:
            all_entries = []
            for item in dataset:
                schematism_page_data = getattr(item, source)
                if schematism_page_data is not None:
                    all_entries.extend(schematism_page_data.entries)
                else:
                    self.logger.warning(f"There is no {source} in this sample. Skipping for global filling.")
            
            # Process all entries across samples sequentially
            current_deanery: Optional[str] = None
            for entry in all_entries:
                if entry.deanery and not current_deanery:
                    current_deanery = entry.deanery
                elif not entry.deanery and current_deanery:
                    entry.deanery = current_deanery
                elif entry.deanery != current_deanery:
                    current_deanery = entry.deanery
                else:
                    pass

        return dataset


class EntriesParsingStep(DatasetProcessingStep[list[PipelineData], list[PipelineData]]):

    def __init__(self, parser: Parser | None = None):
        super().__init__()
        self.parser = parser or Parser()

    def process_dataset(self, dataset: list[PipelineData]) -> list[PipelineData]:
        for item in dataset:
            if item.llm_prediction is not None:
               item.parsed_prediction = self.parser.parse_page(page_data=item.llm_prediction)
        return dataset

class ReplaceValuesStep(DatasetProcessingStep[list[PipelineData], list[PipelineData]]):
    """Replace specific values in SchematismPage entries with new values."""

    def __init__(
        self,
        source: str,
        old_value: str,
        new_value: Any,
        field: str | None = None
    ):
        super().__init__()
        self.source = source
        self.field = field
        self.old_value = old_value
        self.new_value = new_value

    def process_dataset(self, dataset: list[PipelineData]) -> list[PipelineData]:
        for item in dataset:
            page_data = getattr(item, self.source, None)

            if not page_data:
                continue

            if not isinstance(page_data, SchematismPage):
                raise ValueError(
                    f"Source field '{self.source}' is not a SchematismPage object on {item}."
                )

            page_data_dump = page_data.model_dump()

            for entry in page_data_dump.get("entries", []):
                if self.field is not None:
                    # Replace only in the given field
                    if entry.get(self.field) == self.old_value:
                        entry[self.field] = self.new_value
                else:
                    # Replace in all fields
                    for key in list(entry.keys()):
                        if entry[key] == self.old_value:
                            entry[key] = self.new_value

            setattr(item, self.source, SchematismPage(**page_data_dump))

        return dataset







        
        return dataset