"""Use case for updating HuggingFace dataset with generated source (Latin) data."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import final, override, cast, TypedDict

from datasets import Dataset  # pyright: ignore[reportMissingTypeStubs]

from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)
from notarius.schemas.data.dataset import (
    SourceDatasetGenerationResult,
    SchematismPage,
    SchematismsDatasetItem,
)
from notarius.shared.logger import get_logger

logger = get_logger(__name__)

DatasetUpdatesStats = TypedDict(
    "DatasetUpdatesStats",
    {
        "total_samples": int,
        "samples_matched": int,
        "samples_updated": int,
        "samples_unchanged": int,
        "samples_not_found": int,
        "updated_keys": list[str],
        "unchanged_keys": list[str],
    },
)


@dataclass
class UpdateHFWithSourceRequest(BaseRequest):
    """Request to update HuggingFace dataset with generated source data."""

    # Paths to the generated source JSON files (exported from source generation pipeline)
    source_json_paths: list[Path]
    # HuggingFace dataset to update (loaded dataset object)
    hf_dataset: Dataset
    # HuggingFace repo path for pushing (e.g., "username/dataset-name")
    hf_repo_path: str
    # Whether to push the updated dataset to HuggingFace Hub
    push_to_hub: bool = True
    # Whether to create a backup before updating
    create_backup: bool = False
    # Backup path if create_backup is True
    backup_path: Path | None = None
    # Commit message
    commit_message: str | None = None


@dataclass
class UpdateHFWithSourceResponse(BaseResponse):
    """Response from updating HuggingFace dataset with source data."""

    # Statistics about the dataset updates
    stats: DatasetUpdatesStats
    # Whether push to hub was successful
    pushed_to_hub: bool


@final
class UpdateHFDatasetWithSource(
    BaseUseCase[UpdateHFWithSourceRequest, UpdateHFWithSourceResponse]
):
    """
    Use case for updating a HuggingFace dataset with generated source (Latin) data.

    This use case:
    1. Loads generated source data from JSON files (exported from source generation)
    2. Matches source records to HuggingFace dataset samples by schematism_name + filename
    3. Updates the 'source' column in the dataset
    4. Optionally pushes the updated dataset to HuggingFace Hub

    Accepts multiple source JSON files to allow batch updates from multiple schematisms.
    """

    @staticmethod
    def _make_lookup_key(schematism_name: str | None, filename: str | None) -> str:
        """Create a lookup key from schematism_name and filename."""
        return f"{schematism_name or ''}::{filename or ''}"

    @staticmethod
    def _load_source_json_data(source_json_path: Path) -> SourceDatasetGenerationResult:
        if not source_json_path.exists():
            raise FileNotFoundError(f"Source JSON file not found: {source_json_path}")

        with open(source_json_path, "r", encoding="utf-8") as f:
            source_data = cast(SourceDatasetGenerationResult, json.load(f))
            return source_data

    @override
    async def execute(
        self, request: UpdateHFWithSourceRequest
    ) -> UpdateHFWithSourceResponse:
        """
        Execute the HuggingFace dataset update workflow.

        Args:
            request: Request containing source JSON path and HF dataset

        Returns:
            Response with updated statistics and status
        """
        source_by_key: dict[str, SchematismPage] = {}
        total_records_loaded = 0

        for source_json_path in request.source_json_paths:
            logger.info("Loading source data from JSON", path=str(source_json_path))
            source_data = self._load_source_json_data(source_json_path)
            records = source_data.get("records", [])
            total_records_loaded += len(records)

            for record in records:
                metadata = record.get("metadata", {})
                schematism_name = metadata.get("schematism_name")
                filename = metadata.get("filename")

                if schematism_name and filename:
                    lookup_key = self._make_lookup_key(schematism_name, filename)
                    predictions = record.get("predictions", {})
                    source_by_key[lookup_key] = {
                        "page_number": predictions.get("page_number"),
                        "entries": predictions.get("entries", []),
                    }

        logger.info(
            "Loaded source records",
            total_files=len(request.source_json_paths),
            total_records=total_records_loaded,
            indexed_records=len(source_by_key),
        )

        dataset_updates_stats: DatasetUpdatesStats = {
            "total_samples": len(request.hf_dataset),
            "samples_matched": 0,
            "samples_updated": 0,
            "samples_unchanged": 0,
            "samples_not_found": 0,
            "updated_keys": [],
            "unchanged_keys": [],
        }

        def attach_source(sample: SchematismsDatasetItem):
            nonlocal dataset_updates_stats

            sample_schematism_name = sample.get("schematism_name")
            sample_filename = sample.get("filename")

            sample_lookup_key = UpdateHFDatasetWithSource._make_lookup_key(
                sample_schematism_name, sample_filename
            )

            if sample_lookup_key in source_by_key:
                dataset_updates_stats["samples_matched"] += 1

                new_source = source_by_key[sample_lookup_key]
                existing_source = sample.get("source")

                if existing_source == new_source:
                    dataset_updates_stats["samples_unchanged"] += 1
                    dataset_updates_stats["unchanged_keys"].append(sample_lookup_key)
                else:
                    sample["source"] = new_source
                    dataset_updates_stats["samples_updated"] += 1
                    dataset_updates_stats["updated_keys"].append(sample_lookup_key)

            return sample

        logger.info("Updating dataset with source data")
        updated_dataset = (
            request.hf_dataset.map(  # pyright: ignore[reportUnknownMemberType]
                attach_source, load_from_cache_file=False
            )
        )

        dataset_updates_stats["samples_not_found"] = (
            len(request.hf_dataset) - dataset_updates_stats["samples_matched"]
        )

        source_files_desc = (
            request.source_json_paths[0].name
            if len(request.source_json_paths) == 1
            else f"{len(request.source_json_paths)} files"
        )
        default_commit_message = f"update(source): new: {dataset_updates_stats['samples_updated']}, source: {source_files_desc}"

        pushed_to_hub = False
        if request.push_to_hub:
            logger.info(
                "Pushing updated dataset to HuggingFace Hub", repo=request.hf_repo_path
            )
            _ = updated_dataset.push_to_hub(
                repo_id=request.hf_repo_path,
                commit_message=request.commit_message or default_commit_message,
            )
            pushed_to_hub = True
            logger.info("Successfully pushed dataset to HuggingFace Hub")

        return UpdateHFWithSourceResponse(
            stats=dataset_updates_stats,
            pushed_to_hub=pushed_to_hub,
        )
