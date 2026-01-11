"""
Use case for exporting pandas DataFrames to Excel files.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import final, override

import pandas as pd

from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)


@dataclass
class ExcelExportRequest(BaseRequest):
    """Request for exporting a DataFrame to Excel."""

    dataframe: pd.DataFrame
    output_dir: Path
    file_name: str = "predictions.xlsx"
    group_by_key: str = "schematism_name"
    include_index: bool = False
    include_header: bool = True
    add_timestamp: bool = True


@dataclass
class ExcelExportResponse(BaseResponse):
    """Response containing export results."""

    file_path: Path
    sheets_written: list[str] = field(default_factory=list)
    total_rows: int = 0


@final
class ExcelExportUseCase(BaseUseCase[ExcelExportRequest, ExcelExportResponse]):
    """Export pandas DataFrame to Excel file, grouped by a key column.

    Each unique value in the group_by_key column becomes a separate sheet.
    """

    @override
    def execute(self, request: ExcelExportRequest) -> ExcelExportResponse:
        if request.add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{timestamp}_{request.file_name}"
        else:
            file_name = request.file_name

        file_path = request.output_dir / file_name
        sheets_written: list[str] = []

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            for key, group in request.dataframe.groupby(request.group_by_key):
                sheet_name = str(key)
                group.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=request.include_index,
                    header=request.include_header,
                )
                sheets_written.append(sheet_name)

        return ExcelExportResponse(
            file_path=file_path,
            sheets_written=sheets_written,
            total_rows=len(request.dataframe),
        )
