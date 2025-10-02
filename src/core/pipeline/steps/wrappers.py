from __future__ import annotations

from abc import abstractmethod
from typing import Iterator
from typing import List, Optional, Tuple, Dict, Any, Iterable

import pandas as pd

from core.pipeline.steps.base import DatasetProcessingStep, IngestionProcessingStep
from core.pipeline.steps.export import PreferredSource, SaveDataFrameStep
from core.pipeline.steps.ingestion import HuggingFaceIngestionStep
from core.schemas.data.pipeline import PipelineData, PageDataSourceField
from core.schemas.data.schematism import SchematismPage, SchematismEntry


class IngestionStepWrapper[InT, OutT, StepT: IngestionProcessingStep[Any]](
    IngestionProcessingStep[OutT]
):
    """
    A generic wrapper that applies a transformation to each item from a wrapped
    ingestion step on the fly.
    """

    def __init__(self, wrapped_step: StepT, *args, **kwargs):
        """
        Initializes the TransformationWrapper.

        :param wrapped_step: An instance of an IngestionProcessingStep to be wrapped.
        """
        super().__init__(*args, **kwargs)
        self.wrapped_step = wrapped_step

    @abstractmethod
    def transform(self, data: InT) -> OutT:
        """
        Abstract method to transform a single data item.
        Subclasses must implement this to define their specific transformation logic.

        :param data: The data item from the wrapped step.
        :return: The transformed data item.
        """
        pass

    @abstractmethod
    def iter_source(self) -> Iterator[OutT]:
        """
        Iterates over the wrapped step's source, applies the transformation to each item,
        and yields the transformed result.
        """
        ...


class DatasetProcessingStepWrapper[PassT, TransformT, StepT: DatasetProcessingStep](
    DatasetProcessingStep[PassT, PassT]
):
    def __init__(self, wrapped_step: StepT):
        super().__init__()
        self.wrapped_step = wrapped_step

    @abstractmethod
    def transform(self, dataset: PassT) -> TransformT:
        pass

    def process_dataset(self, dataset: PassT) -> PassT:

        self.wrapped_step.process_dataset(self.transform(dataset))

        return dataset


class HuggingFaceToPipelineDataStep(
    IngestionStepWrapper[dict, PipelineData, HuggingFaceIngestionStep]
):

    def __init__(self, wrapped_step: HuggingFaceIngestionStep, column_map: dict):
        super().__init__(wrapped_step=wrapped_step)
        column_map = column_map

        self.image_src_col = column_map.get("image_column")
        self.ground_truth_src_col = column_map.get("ground_truth_column")

        if not self.image_src_col or not self.ground_truth_src_col:
            raise ValueError(
                "Config's column_map must specify 'image_column' and 'ground_truth_column'."
            )

    def iter_source(self) -> Iterator[PipelineData]:
        for item in self.wrapped_step.iter_source():
            yield self.transform(item)

    def transform(self, data: dict) -> PipelineData:

        image = data.get(self.image_src_col)
        parsed_ground_truth = data.get(self.ground_truth_src_col)
        source_ground_truth = data.get("source")

        if image is None or parsed_ground_truth is None:
            raise ValueError(
                f"Missing required data. Looking for '{self.image_src_col}' and '{self.ground_truth_src_col}' in the data sample."
            )

        metadata = {
            "schematism": data.get("schematism_name"),
            "filename": data.get("filename"),
        }

        mapped_data = {
            "image": image,
            "ground_truth": SchematismPage(**parsed_ground_truth),
            "source_ground_truth": SchematismPage(**source_ground_truth),
            "metadata": metadata,
        }

        return PipelineData(**mapped_data)


class PipelineDataToPandasDataFrameStep(
    DatasetProcessingStepWrapper[
        Iterable[PipelineData], pd.DataFrame, SaveDataFrameStep
    ]
):
    """Converts a list of `PipelineData` into a flat pandas ``DataFrame``.

    Each row corresponds to an individual parish entry. Page-level attributes and
    selected metadata keys are propagated to each row.

    Produces the following columns when available:
      - sample_index: Index of the `PipelineData` item in the input list.
      - source: Which source object was used (``parsed`` | ``llm`` | ``lmv3`` | ``ground_truth``).
      - page_number: Page number from the selected source.
      - parish, deanery, dedication, building_material: Entry-level fields.
      - metadata fields: Included as top-level columns using their original keys.
    """

    def __init__(
        self,
        wrapped_step: SaveDataFrameStep,
        source: PageDataSourceField,
        include_metadata: bool = True,
        metadata_keys: List[str] | None = None,
    ) -> None:
        """Initializes the step.

        Args:
            source: Source to use for rows. When ``"auto"``, selects the first available
                among ``parsed`` → ``llm`` → ``lmv3`` → ``ground_truth``.
            include_metadata: Whether to include metadata fields as columns.
            metadata_keys: Optional allowlist of metadata keys to include. When ``None``
                and ``include_metadata=True``, all metadata keys found are included.
        """
        super().__init__(wrapped_step=wrapped_step)
        self.source = source
        self.include_metadata = include_metadata
        self.metadata_keys = metadata_keys

    def _metadata_columns(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Builds a dictionary of metadata columns based on configuration.

        Args:
            metadata: The metadata mapping from the sample.

        Returns:
            A mapping of column description to value to merge into the output row.
        """
        if not self.include_metadata:
            return {}
        if self.metadata_keys is None:
            return dict(metadata)
        return {k: metadata.get(k) for k in self.metadata_keys}

    def _entry_to_row(
        self,
        sample_index: int,
        source_name: str,
        page: SchematismPage,
        entry: SchematismEntry,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Converts a single entry to a flat row dictionary.

        Args:
            sample_index: Index of the sample in the data list.
            source_name: Name of the selected source.
            page: Page-level data selected for this sample.
            entry: Entry-level data to flatten into a row.
            metadata: Metadata mapping associated with the sample.

        Returns:
            A flat row ready to be appended to the output ``DataFrame``.
        """
        row: Dict[str, Any] = {
            "sample_index": sample_index,
            "source": source_name,
            "page_number": page.page_number,
            "parish": entry.parish,
            "deanery": entry.deanery,
            "dedication": entry.dedication,
            "building_material": entry.building_material,
        }

        row.update(self._metadata_columns(metadata))

        return row

    def transform(self, dataset) -> pd.DataFrame:
        """Builds a flat DataFrame from a list of ``PipelineData`` samples.

        Args:
           self: abc
           dataset: The input list of pipeline samples to convert.
           **kwargs: Unused.

        Returns:
           A pandas ``DataFrame`` with one row per parish entry across all samples.
           If no entries are found, an empty ``DataFrame`` with expected columns is returned.
        """
        rows: List[Dict[str, Any]] = []

        for idx, item in enumerate(dataset):
            page = getattr(item, self.source, None)
            if page is None or not page.entries:
                continue

            for entry in page.entries:
                rows.append(
                    self._entry_to_row(
                        sample_index=idx,
                        source_name=self.source,
                        page=page,
                        entry=entry,
                        metadata=item.metadata or {},
                    )
                )

        if not rows:
            # Return an empty DataFrame with expected columns for consistency
            columns = pd.Index(
                [
                    "sample_index",
                    "source",
                    "page_number",
                    "parish",
                    "deanery",
                    "dedication",
                    "building_material",
                ]
            )
            # Include metadata keys if requested but unavailable yet
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame.from_records(rows)

        df = df.sort_values(by=["sample_index"])

        return df


class DataFrameSchemaMappingStep(DatasetProcessingStep[pd.DataFrame, pd.DataFrame]):
    """Maps/renames DataFrame columns and optionally enforces a target schema.

    This step is intended to prepare a pandas ``DataFrame`` for database export
    by renaming columns according to a provided mapping and, if desired,
    restricting the output to a specific set of target columns in a defined
    order.

    Example:
        mapping = {"parish": "parafia", "deanery": "dekanat"}
        use_columns = ["id", "parafia", "dekanat", "skany"]
        step = DataFrameSchemaMappingStep(mapping=mapping, use_columns=use_columns)
        export_df = step.process(df)

    Attributes:
        mapping: Mapping from input column names to output column names.
        target_columns: Optional ordered list of columns to enforce on the output
            DataFrame. Missing columns are created and filled with ``NaN``.
        strict: When ``True``, raise if any source column in ``mapping`` is not
            present in the input DataFrame.
        preserve_unmapped: When no ``use_columns`` is provided, controls whether
            columns not mentioned in ``mapping`` are preserved in the output.
            Defaults to ``True``.
    """

    def __init__(
        self,
        mapping: Dict[str, str],
        target_columns: Optional[List[str]] = None,
        strict: bool = False,
        preserve_unmapped: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the schema mapping step.

        Args:
            mapping: Mapping from input column names (keys) to output names (values).
            target_columns: Optional ordered list of desired output columns.
            strict: If ``True``, raise ``ValueError`` when a mapping source column
                is missing in the input DataFrame.
            preserve_unmapped: If no ``use_columns`` is provided, whether to
                keep columns not present in ``mapping``.

        Raises:
            ValueError: If ``mapping`` is empty or contains duplicate destination names.
        """

        super().__init__()

        if not mapping:
            raise ValueError("mapping must not be empty")

        dest_names = list(mapping.values())
        if len(dest_names) != len(set(dest_names)):
            raise ValueError("mapping contains duplicate destination column names")

        self.mapping = mapping
        self.target_columns = target_columns
        self.strict = strict
        self.preserve_unmapped = preserve_unmapped

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames columns using the mapping while honoring strict mode.

        Args:
            df: Input DataFrame.

        Returns:
            A new DataFrame with columns renamed when possible.

        Raises:
            ValueError: If ``strict`` is ``True`` and a source column is missing.
        """
        missing_sources = [src for src in self.mapping.keys() if src not in df.columns]
        if missing_sources and self.strict:
            raise ValueError(f"Missing source columns for mapping: {missing_sources}")

        # Only rename those that exist; absent sources are ignored in non-strict mode
        effective_mapping = {
            src: dst for src, dst in self.mapping.items() if src in df.columns
        }
        return df.rename(columns=effective_mapping)

    def _enforce_target_schema(self, df_renamed: pd.DataFrame) -> pd.DataFrame:
        """Creates an output DataFrame limited to ``use_columns`` in order.

        Args:
            df_renamed: DataFrame after renaming.

        Returns:
            DataFrame with exactly ``use_columns`` in the specified order. Any
            columns not present in ``df_renamed`` are created with ``NaN`` values.
        """
        assert self.target_columns is not None

        out = pd.DataFrame(
            index=df_renamed.index, columns=pd.Index(self.target_columns)
        )
        for col in self.target_columns:
            if col in df_renamed.columns:
                out[col] = df_renamed[col]
            # else leave as NaN
        return out

    def process_dataset(self, dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Applies column mapping and optional schema enforcement to a DataFrame.

        Args:
            dataset: Input DataFrame to transform.
            **kwargs: Unused.

        Returns:
            Transformed DataFrame where columns are renamed and optionally
            constrained to ``use_columns``.
        """
        df = dataset.copy()
        df_renamed = self._rename_columns(df)

        if self.target_columns is not None:
            return self._enforce_target_schema(df_renamed)

        if self.preserve_unmapped:
            return df_renamed

        # Keep only destination columns that came from mapping
        kept_columns = set(self.mapping.values())
        existing_kept = [c for c in df_renamed.columns if c in kept_columns]
        return df_renamed.loc[:, existing_kept]
