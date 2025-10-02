"""Export and serialization steps for pipeline datasets.

This module provides pipeline steps that transform pipeline data structures
into external representations such as pandas DataFrames, and utilities to
persist those DataFrames to files.
"""

from __future__ import annotations

import json
import os
import random
import re
import sqlite3
from typing import Literal, Optional, Callable, Any, Sequence

import pandas as pd
from pathlib import Path

import sqlalchemy
from sqlalchemy import Engine

from core.pipeline.steps.base import DatasetProcessingStep
from core.schemas.data.pipeline import PipelineData
from core.schemas.data.schematism import SchematismPage
from core.utils.shared import TMP_DIR, DATA_DIR

PreferredSource = Literal["auto", "parsed", "llm", "lmv3", "ground_truth"]
SaveFormat = Literal["csv", "excel"]


class SaveDataFrameStep(DatasetProcessingStep[pd.DataFrame, pd.DataFrame]):
    """Saves a pandas ``DataFrame`` to disk in CSV or Excel format.

    This step writes the provided DataFrame to the given file path and returns
    the DataFrame unchanged, enabling further processing if needed.

    Example:
        step = SaveDataFrameStep("/tmp/output.csv", format="csv", overwrite=True)
        df = step.process(df)

    Attributes:
        file_path: Destination file path.
        file_format: Output format, either ``"csv"`` or ``"excel"``. If omitted, the
            format is inferred from the ``file_path`` extension (``.csv`` or ``.xlsx``/``.xls``).
        overwrite: Whether to overwrite an existing file.
        include_index: Whether to include the DataFrame index in the output file.
        excel_sheet_name: Optional sheet description for Excel output.
    """

    def __init__(
        self,
        file_format: SaveFormat,
        group_by_metadata_key: str,
        file_name: str = "predictions",
        overwrite: bool = True,
        include_index: bool = True,
        excel_sheet_name: str | None = None,
    ) -> None:
        """Initializes the saver step.

        Args:
            file_path: Destination file path.
            file_format: Output file_format. If ``None``, inferred from the file extension.
            overwrite: If ``True``, allow overwriting an existing file.
            include_index: If ``True``, include the DataFrame index in the file.
            excel_sheet_name: Optional sheet description for Excel output; used when
                file_format is ``"excel"``.

        Raises:
            ValueError: If the file_format cannot be inferred or is unsupported.
        """
        super().__init__()
        self.file_format = file_format
        self.group_by_metadata_key = group_by_metadata_key
        self.file_name = file_name
        self.overwrite = overwrite
        self.include_index = include_index
        self.excel_sheet_name = excel_sheet_name

    def _infer_format(self) -> SaveFormat:
        """Infers the output file_format based on the ``file_path`` suffix.

        Returns:
            The inferred save file_format.

        Raises:
            ValueError: If the suffix is not recognized as CSV or Excel.
        """
        suffix = self.file_path.suffix.lower()
        if suffix == ".csv":
            return "csv"
        if suffix in (".xlsx", ".xls"):
            return "excel"
        raise ValueError(f"Cannot infer save file_format from extension: {suffix}")

    def process_dataset(self, dataset: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Writes the DataFrame to file and returns it unchanged.

        Args:
            dataset: The DataFrame to persist to disk.
            **kwargs: Unused.

        Returns:
            The same DataFrame instance that was provided.

        Raises:
            FileExistsError: If the destination exists and ``overwrite`` is ``False``.
            ValueError: If the desired or inferred file_format is unsupported.
        """
        out_format: SaveFormat = self.file_format or self._infer_format()

        self.logger.info(f"Number of samples in dataset {len(dataset)}")

        if out_format == "csv":
            file_path = (DATA_DIR / self.file_name).with_suffix(".xlsx")
            for key, group in dataset.groupby(self.group_by_metadata_key):
                dataset.to_excel(f"{file_path}_{key}", index=self.include_index)

        elif out_format == "excel":

            file_path = (DATA_DIR / self.file_name).with_suffix(".xlsx")

            if file_path.exists():
                if self.overwrite:
                    os.remove(file_path)
                else:
                    raise FileExistsError(
                        f"File already exists: {file_path}, use overwrite=True to overwrite."
                    )

            with pd.ExcelWriter(file_path) as writer:
                for key, group in dataset.groupby(self.group_by_metadata_key):
                    group.to_excel(writer, sheet_name=key, header=self.include_index)
        else:
            raise ValueError(f"Unsupported save file_format: {out_format}")

        return dataset


class AppendDataFrameToSQLStep(DatasetProcessingStep[pd.DataFrame, pd.DataFrame]):
    """Appends a pandas ``DataFrame`` to a SQL table.

    Designed for PostgreSQL via SQLAlchemy URL (e.g.,
    ``postgresql+psycopg2://user:pass@host:port/dbname``) but compatible with any
    SQLAlchemy-supported backend or DB-API connection (e.g., SQLite).

    Attributes:
        table_name: Destination table name to append into.
        connection: SQLAlchemy URL string, SQLAlchemy Engine, or DB-API connection.
        if_exists: Behavior if the table exists: one of ``"append"``, ``"fail"``,
            or ``"replace"`` (passed to ``DataFrame.to_sql``). Defaults to ``"append"``.
        index: Whether to write the DataFrame index as a column.
        chunksize: Optional chunk size for batch inserts.
        method: Insert method for ``to_sql`` (e.g., ``None``, ``"multi"`, or callable).
        dtype: Optional column type mapping for SQLAlchemy.
    """

    def __init__(
        self,
        table_name: str,
        connection: Engine | sqlite3.Connection | str,
        if_exists: Literal["fail", "replace", "append"] = "append",
        index: bool = False,
        chunksize: Optional[int] = None,
        method: None | Literal["multi"] | Callable[..., int | None] = None,
        dtype: Any | None = None,
        use_columns: Sequence[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initializes the SQL append step.

        Args:
            table_name: Name of the SQL table.
            connection: SQLAlchemy URL/Engine or DB-API connection.
            if_exists: Behavior when table exists. Defaults to ``"append"``.
            index: Whether to include DataFrame index in SQL table.
            chunksize: Chunk size for batch inserts.
            method: Insert method for ``to_sql``.
            dtype: Optional type mapping for columns.
        """
        super().__init__(*args, **kwargs)
        self.table_name = table_name
        self.connection = connection
        self.if_exists = if_exists
        self.index = index
        self.chunksize = chunksize
        self.method = method
        self.dtype = dtype
        self.use_columns = use_columns

    def _resolve_connection(self):
        """Resolves the provided connection into a SQLAlchemy engine or DB-API connection.

        Returns:
            An object accepted by ``pandas.DataFrame.to_sql`` for the ``con`` parameter.

        Raises:
            ValueError: If the connection type is unsupported or SQLAlchemy is required but missing.
        """
        con = self.connection
        # SQLAlchemy URL string
        if isinstance(con, str):
            return sqlalchemy.create_engine(con)
        elif isinstance(con, Engine):
            return con
        elif isinstance(con, sqlite3.Connection):
            return con

        # Unsupported connection type
        raise ValueError(f"Unsupported connection type: {type(con)}")

    def process_dataset(self, dataset: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Appends the DataFrame to the configured SQL table and returns it unchanged.

        Args:
            dataset: The DataFrame to append into the SQL table.
            **kwargs: Unused.

        Returns:
            The same DataFrame instance.
        """

        assert self.if_exists in (
            "fail",
            "replace",
            "append",
        ), f"Invalid if_exists: {self.if_exists}"
        assert self.method in (None, "multi"), f"Invalid method: {self.method}"

        con = self._resolve_connection()

        target_table = pd.read_sql_query(f"SELECT * FROM {self.table_name}", con)
        target_table_columns = target_table.columns.tolist()

        dataset_subset = dataset[self.use_columns]

        missing_columns = []
        for column_name in dataset_subset.columns:
            if column_name not in target_table_columns:
                missing_columns.append(column_name)
        if missing_columns:
            raise ValueError(
                f"Column {missing_columns} not found in table {self.table_name}"
            )

        dataset_subset.to_sql(
            name=self.table_name,
            con=con,
            if_exists=self.if_exists,
            index=self.index,
            chunksize=self.chunksize,
            method=self.method,
            dtype=self.dtype,
        )

        return dataset


class DownloadSamplesStep(
    DatasetProcessingStep[list[PipelineData], list[PipelineData]]
):
    """Downloads samples to a specified directory.

    This step iterates through a list of ``PipelineData`` objects and saves the
    image and ground truth data for each sample to separate files. The filenames
    are derived from the ``file_name`` metadata attribute.

    Example:
        A sample with ``metadata["file_name"] = "0043.tif"`` will be saved as:
        - ``/path/to/directory/0043.jpg`` (image)
        - ``/path/to/directory/0043.json`` (ground truth)

    Attributes:
        directory: The directory where the samples will be saved.
    """

    def __init__(
        self,
        directory: Path,
        samples_num: int | None = None,
        shuffle: bool = False,
        schematism_name_column: str = "schematism",
        file_name_column: str = "filename",
    ):
        """Initializes the DownloadSamplesStep.

        Args:
            directory: The directory where the samples will be saved.
            samples_num: The number of samples to download. If None, all samples
                are downloaded.
            shuffle: Whether to shuffle the dataset before downloading.
        """
        super().__init__()
        self.directory = directory
        self.samples_num = samples_num
        self.shuffle = shuffle
        self.schematism_name_column = schematism_name_column
        self.file_name_column = file_name_column
        self.directory.mkdir(parents=True, exist_ok=True)

    def process_dataset(self, dataset: list[PipelineData]) -> list[PipelineData]:
        """Saves the image and ground truth for each sample in the dataset.

        Args:
            dataset: A list of ``PipelineData`` objects to process.

        Returns:
            The input dataset, unchanged.
        """

        if self.shuffle:
            random.shuffle(dataset)

        dataset_to_process = (
            dataset[: self.samples_num] if self.samples_num is not None else dataset
        )
        for sample in dataset_to_process:
            self.logger.info(f"Processing sample {sample.metadata}")
            if self.file_name_column not in sample.metadata:
                self.logger.warning(
                    "Skipping sample because 'file_name' is missing from metadata.",
                    metadata=sample.metadata,
                )
                continue

            base_name = Path(
                f"{sample.metadata[self.schematism_name_column]}_{sample.metadata[self.file_name_column]})"
            ).stem

            if sample.image:
                image_path = self.directory / f"{base_name}.jpg"
                sample.image.save(image_path)
                self.logger.info(f"Saved image to {image_path}")
            else:
                self.logger.warning("Skipping image because it is missing.")

            if sample.ground_truth:
                gt_path = self.directory / f"{base_name}.json"
                with open(gt_path, "w", encoding="utf-8") as f:
                    f.write(sample.ground_truth.model_dump_json(indent=4))
                self.logger.info(f"Saved ground truth to {gt_path}")
            else:
                self.logger.warning("Skipping ground truth because it is missing.")

        return dataset


class SaveJSONStep(DatasetProcessingStep[list[PipelineData], list[PipelineData]]):
    """Persists per-sample pipeline data as JSON files.

    The step selects a ``SchematismPage``-like object from each ``PipelineData``
    item according to the configured ``source`` attribute and writes it to disk
    as JSON. Filenames are derived from the sample metadata using the pattern
    ``"{schematism_name}_{file_name}.json"`` (sanitized for filesystem safety).

    Attributes:
        directory: Destination directory where JSON files will be saved.
        source: Preferred data source to serialize (``parsed`` | ``llm`` |
            ``lmv3`` | ``ground_truth`` | ``auto``).
        overwrite: Whether existing files may be replaced.
        indent: Indentation level passed to :func:`json.dump` for readability.
        ensure_ascii: Whether to escape non-ASCII characters in JSON output.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        source: str = "llm_prediction",
        overwrite: bool = True,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> None:
        super().__init__()

        self.directory = (
            Path(directory)
            if directory is not None
            else TMP_DIR / "generated" / "dataset"
        )
        self.source: PreferredSource = source
        self.overwrite = overwrite
        self.indent = indent
        self.ensure_ascii = ensure_ascii

        self.directory.mkdir(parents=True, exist_ok=True)

    def process_dataset(
        self, dataset: list[PipelineData], **kwargs: Any
    ) -> list[PipelineData]:
        for sample in dataset:
            page_data = getattr(sample, self.source, None)

            if page_data is None:
                self.logger.warning(
                    "Skipping sample because no data found for configured source.",
                    configured_source=self.source,
                    metadata=sample.metadata,
                )
                continue
            if not isinstance(page_data, SchematismPage):
                raise TypeError("Selected source is not a SchematismPage object.")

            schematism_name = sample.metadata["schematism"]
            file_name = sample.metadata["filename"]

            if schematism_name is None or file_name is None:
                self.logger.warning(
                    "Skipping sample because required metadata keys are missing.",
                )
                continue

            base_name = Path(f"{schematism_name}_{file_name}").stem
            output_path = (self.directory / base_name).with_suffix(".json")

            if output_path.exists() and not self.overwrite:
                self.logger.warning(
                    "Skipping sample because destination file already exists and overwrite is disabled.",
                    path=str(output_path),
                )
                continue

            payload = page_data.model_dump()

            with open(output_path, "w", encoding="utf-8") as fh:
                json.dump(
                    payload, fh, indent=self.indent, ensure_ascii=self.ensure_ascii
                )

        return dataset
