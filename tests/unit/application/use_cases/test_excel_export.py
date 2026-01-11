"""Tests for ExcelExportUseCase."""

import pytest
import pandas as pd
from pathlib import Path

from notarius.application.use_cases.export.excel_export import (
    ExcelExportUseCase,
    ExcelExportRequest,
)


class TestExcelExportUseCase:
    """Test suite for ExcelExportUseCase."""

    def test_execute_creates_excel_file(self, tmp_path: Path) -> None:
        """Test that the use case creates an Excel file with multiple sheets."""
        df = pd.DataFrame(
            {
                "schematism_name": ["A", "A", "B"],
                "col1": [1, 2, 3],
            }
        )

        use_case = ExcelExportUseCase()
        request = ExcelExportRequest(
            dataframe=df,
            output_dir=tmp_path,
            file_name="test.xlsx",
            add_timestamp=False,
        )

        response = use_case.execute(request)

        assert response.file_path.exists()
        assert response.total_rows == 3
        assert set(response.sheets_written) == {"A", "B"}

        # Verify file content
        xl = pd.ExcelFile(response.file_path)
        assert set(xl.sheet_names) == {"A", "B"}

        df_a = xl.parse("A")
        assert len(df_a) == 2

        df_b = xl.parse("B")
        assert len(df_b) == 1

    def test_execute_with_timestamp(self, tmp_path: Path) -> None:
        """Test that timestamp is added to filename if requested."""
        df = pd.DataFrame({"schematism_name": ["A"], "col1": [1]})
        use_case = ExcelExportUseCase()
        request = ExcelExportRequest(
            dataframe=df,
            output_dir=tmp_path,
            file_name="test.xlsx",
            add_timestamp=True,
        )

        response = use_case.execute(request)

        assert response.file_path.name.endswith("_test.xlsx")
        assert len(response.file_path.name) > len("test.xlsx")
