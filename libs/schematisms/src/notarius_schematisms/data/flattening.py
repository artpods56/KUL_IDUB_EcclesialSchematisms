from typing import Any

from notarius_schematisms.domain.dataset import (
    PredictionDataItem,
    BaseMetaData,
    FlatSchematismEntryWithMetadata,
    FlatSchematismAlignedEntryWithMetadata,
    AlignedSchematismsDataItem,
)


def prepare_metadata(metadata: BaseMetaData | None) -> BaseMetaData:
    return (
        metadata
        if metadata
        else BaseMetaData(sample_id=0, filename="unknown", schematism_name="unknown")
    )


class FlatteningService:
    # Define preferred column order for exports
    PREDICTION_COLUMN_ORDER = [
        "sample_id",
        "schematism_name",
        "filename",
        "deanery",
        "parish",
        "dedication",
        "building_material",
    ]

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
    def to_ordered_dicts(
        items: list[FlatSchematismEntryWithMetadata],
    ) -> list[dict[str, Any]]:
        """Convert flattened items to ordered dictionaries with metadata columns first.

        This ensures consistent column ordering in DataFrames and Excel exports,
        with metadata fields (sample_id, schematism_name, filename) appearing first.
        """
        ordered_dicts = []
        for item in items:
            data = item.model_dump()
            # Reorder dictionary to match preferred column order
            ordered_dict = {
                key: data[key]
                for key in FlatteningService.PREDICTION_COLUMN_ORDER
                if key in data
            }
            # Add any remaining fields that weren't in the preferred order
            for key, value in data.items():
                if key not in ordered_dict:
                    ordered_dict[key] = value
            ordered_dicts.append(ordered_dict)
        return ordered_dicts

    # Define preferred column order for aligned (evaluation) exports
    ALIGNED_COLUMN_ORDER = [
        "sample_id",
        "schematism_name",
        "filename",
        "deanery_a",
        "deanery_b",
        "parish_a",
        "parish_b",
        "dedication_a",
        "dedication_b",
        "building_material_a",
        "building_material_b",
    ]

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
                    building_material_a=entry_a.building_material,
                    building_material_b=entry_b.building_material,
                )
            )

        return flat_entries

    @staticmethod
    def to_ordered_dicts_aligned(
        items: list[FlatSchematismAlignedEntryWithMetadata],
    ) -> list[dict[str, Any]]:
        """Convert aligned flattened items to ordered dictionaries with metadata columns first.

        This ensures consistent column ordering in aligned/evaluation DataFrames and Excel exports.
        """
        ordered_dicts = []
        for item in items:
            data = item.model_dump()
            # Reorder dictionary to match preferred column order
            ordered_dict = {
                key: data[key]
                for key in FlatteningService.ALIGNED_COLUMN_ORDER
                if key in data
            }
            # Add any remaining fields that weren't in the preferred order
            for key, value in data.items():
                if key not in ordered_dict:
                    ordered_dict[key] = value
            ordered_dicts.append(ordered_dict)
        return ordered_dicts
