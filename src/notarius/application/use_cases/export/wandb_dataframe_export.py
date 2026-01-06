"""Use case for exporting dataframes to Weights & Biases."""

from dataclasses import dataclass, field
from typing import final, override

import pandas as pd
import wandb
from PIL import Image
from structlog import get_logger

from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)

logger = get_logger(__name__)


@dataclass
class DataFrameExportConfig:
    """Configuration for a single dataframe export."""

    dataframe: pd.DataFrame
    table_name: str
    group_by_key: str | None = None
    include_images: bool = True
    sample_id_column: str = "sample_id"


@dataclass
class WandBExportRequest(BaseRequest):
    """Request to export dataframes to WandB."""

    exports: list[DataFrameExportConfig]
    sample_id_to_image: dict[str, Image.Image] = field(default_factory=dict)


@dataclass
class WandBExportResponse(BaseResponse):
    """Response from WandB export."""

    tables_logged: list[str]
    total_rows: int


@final
class ExportDataFrameToWandB(BaseUseCase[WandBExportRequest, WandBExportResponse]):
    """Use case for exporting one or more dataframes to Weights & Biases.

    This use case handles the logic of converting pandas DataFrames to WandB tables,
    optionally grouping by a key and including images.
    """

    def __init__(self, wandb_run: wandb.sdk.wandb_run.Run):
        """Initialize the use case.

        Args:
            wandb_run: Active WandB run to log tables to
        """
        self.run = wandb_run

    @override
    async def execute(self, request: WandBExportRequest) -> WandBExportResponse:
        """Execute the WandB export workflow.

        Args:
            request: Request containing dataframes and export configurations

        Returns:
            Response with list of logged table names and total row count
        """
        tables_logged: list[str] = []
        total_rows = 0

        for export_config in request.exports:
            logged_tables = self._export_dataframe(
                dataframe=export_config.dataframe,
                table_name=export_config.table_name,
                group_by_key=export_config.group_by_key,
                include_images=export_config.include_images,
                sample_id_column=export_config.sample_id_column,
                sample_id_to_image=request.sample_id_to_image,
            )
            tables_logged.extend(logged_tables)
            total_rows += len(export_config.dataframe)

        logger.info(
            "WandB export completed",
            tables_logged=len(tables_logged),
            total_rows=total_rows,
        )

        return WandBExportResponse(
            tables_logged=tables_logged,
            total_rows=total_rows,
        )

    def _export_dataframe(
        self,
        dataframe: pd.DataFrame,
        table_name: str,
        group_by_key: str | None,
        include_images: bool,
        sample_id_column: str,
        sample_id_to_image: dict[str, Image.Image],
    ) -> list[str]:
        """Export a single dataframe to WandB.

        Returns:
            List of table names that were logged
        """
        tables_logged: list[str] = []

        if group_by_key:
            # Log each group as a separate table
            for key, group in dataframe.groupby(group_by_key):
                full_table_name = f"{table_name}_{key}"
                table = self._build_table(
                    dataframe=group,
                    include_images=include_images,
                    sample_id_column=sample_id_column,
                    sample_id_to_image=sample_id_to_image,
                )
                self.run.log({full_table_name: table})
                tables_logged.append(full_table_name)
                logger.debug(f"Logged table '{full_table_name}' to W&B")
        else:
            # Log the entire dataframe as one table
            table = self._build_table(
                dataframe=dataframe,
                include_images=include_images,
                sample_id_column=sample_id_column,
                sample_id_to_image=sample_id_to_image,
            )
            self.run.log({table_name: table})
            tables_logged.append(table_name)
            logger.debug(f"Logged table '{table_name}' to W&B")

        return tables_logged

    def _build_table(
        self,
        dataframe: pd.DataFrame,
        include_images: bool,
        sample_id_column: str,
        sample_id_to_image: dict[str, Image.Image],
    ) -> wandb.Table:
        """Build a WandB table from a dataframe.

        Args:
            dataframe: DataFrame to convert
            include_images: Whether to include images in the table
            sample_id_column: Column name containing sample IDs for image lookup
            sample_id_to_image: Mapping from sample ID to PIL Image

        Returns:
            WandB Table ready to be logged
        """
        if include_images and sample_id_to_image:
            columns = ["image"] + list(dataframe.columns)
            data = []

            for sample_id, grouped_samples in dataframe.groupby(sample_id_column):
                # Get image for this sample_id
                pil_image = sample_id_to_image.get(sample_id)
                if pil_image:
                    wandb_image = wandb.Image(pil_image.resize((400, 600)))
                else:
                    wandb_image = None

                # Add image only to the first row, None for the rest
                for i, (idx, row) in enumerate(grouped_samples.iterrows()):
                    if i == 0:
                        data.append([wandb_image] + row.tolist())
                    else:
                        data.append([None] + row.tolist())
        else:
            columns = list(dataframe.columns)
            data = dataframe.values.tolist()

        return wandb.Table(columns=columns, data=data)
