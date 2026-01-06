from typing import Any

from notarius.schemas.data.pipeline import (
    PredictionDataItem,
    BaseMetaData,
    FlatSchematismEntryWithMetadata,
    FlatSchematismAlignedEntryWithMetadata,
    AlignedSchematismsDataItem,
)
from notarius.domain.entities.schematism import SchematismEntry, SchematismPage


def prepare_metadata(metadata: BaseMetaData | None) -> BaseMetaData:
    return (
        metadata
        if metadata
        else BaseMetaData(sample_id=0, filename="unknown", schematism_name="unknown")
    )


class FlatteningService:

    @staticmethod
    def flatten_prediction_data_item(
        item: PredictionDataItem,
    ) -> list[FlatSchematismEntryWithMetadata]:

        metadata = prepare_metadata(item.metadata)

        flat_items: list[FlatSchematismEntryWithMetadata] = []

        for entry in item.predictions.entries:

            flat_items.append(
                FlatSchematismEntryWithMetadata(
                    sample_id=metadata.sample_id,
                    filename=metadata.filename,
                    schematism_name=metadata.schematism_name,
                    deanery=entry.deanery,
                    parish=entry.parish,
                    dedication=entry.dedication,
                    building_material=entry.building_material,
                )
            )

        return flat_items

    @staticmethod
    def flatten_aligned_pages(
        item: AlignedSchematismsDataItem,
    ) -> list[FlatSchematismAlignedEntryWithMetadata]:

        metadata = prepare_metadata(item.metadata)

        flat_entries: list[FlatSchematismAlignedEntryWithMetadata] = []

        page_a, page_b = item.aligned_schematism_pages

        for entry_a, entry_b in zip(page_a.entries, page_b.entries):
            flat_entries.append(
                FlatSchematismAlignedEntryWithMetadata(
                    sample_id=metadata.sample_id,
                    schematism_name=metadata.schematism_name,
                    filename=metadata.filename,
                    parish_a=entry_a.parish,
                    parish_b=entry_b.parish,
                    deanery_a=entry_a.deanery,
                    deanery_b=entry_b.deanery,
                    dedication_a=entry_a.dedication,
                    dedication_b=entry_b.dedication,
                )
            )

        return flat_entries
