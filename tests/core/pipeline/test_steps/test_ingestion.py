"""Tests for ingestion pipeline steps.

This module verifies the behavior of `ImageFileIngestionStep` and
`TextFileIngestionStep`, including handling of valid inputs, empty or
non-matching directories, and correctness of produced `PipelineData`
metadata.
"""

from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest
from PIL import Image
from typing import List
from core.pipeline.steps.ingestion import (
    ImageFileExtension,
    ImageFileIngestionStep,
    PDFFileExtension,
    TextFileExtension,
    TextFileIngestionStep,
    PdfFileIngestionStep,
)
from core.schemas.data.pipeline import PipelineData


class TestImageFileIngestionStep:
    """Tests covering image file ingestion behavior."""

    @pytest.fixture
    def mock_directory(self, tmp_path):
        """Return a temporary directory path for image ingestion tests."""
        return tmp_path

    @pytest.fixture
    def mock_image_file(self, mock_directory):
        """Create and return a small test image within the temp directory."""
        image_path = mock_directory / "test_image.jpg"
        img = Image.new("RGB", (60, 30), color="red")
        img.save(image_path)
        return image_path

    @pytest.fixture
    def ingestion_step(self, mock_directory):
        """Instantiate an `ImageFileIngestionStep` for the temp directory."""
        file_extensions: List[ImageFileExtension] = [".jpg", ".png"]
        return ImageFileIngestionStep(
            data_directory=str(mock_directory), file_extensions=file_extensions
        )

    def test_process_with_valid_image(self, ingestion_step, mock_image_file):
        """Process a valid image and produce one `PipelineData` with metadata."""
        with patch(
            "core.pipeline.steps.ingestion.PipelineData", wraps=PipelineData
        ) as mock_pipeline_data:
            result = ingestion_step.process()

            assert len(result) == 1
            assert isinstance(result[0], PipelineData)
            assert mock_pipeline_data.call_count == 1
            assert result[0].metadata["file_name"] == mock_image_file.name

    def test_process_with_no_matching_files(self, ingestion_step, mock_directory):
        """Return an empty result when no image files match configured extensions."""
        unknown_file = mock_directory / "document.pdf"
        unknown_file.touch()

        result = ingestion_step.process()

        assert len(result) == 0

    def test_process_with_invalid_file_extension(self, ingestion_step, mock_directory):
        """Ignore files with invalid extensions and return no results."""
        invalid_file = mock_directory / "invalid_file.txt"
        invalid_file.touch()

        result = ingestion_step.process()

        assert len(result) == 0

    def test_process_with_empty_directory(self, ingestion_step):
        """Return an empty result when the input directory is empty."""
        result = ingestion_step.process()

        assert len(result) == 0

    def test_process_metadata_content(self, ingestion_step, mock_image_file):
        """Validate the metadata contents of the produced `PipelineData`."""
        result = ingestion_step.process()

        expected_metadata = {
            "file_path": str(mock_image_file),
            "file_name": mock_image_file.name,
        }

        assert result[0].metadata == expected_metadata


class TestTextFileIngestionStep:
    """Tests covering text file ingestion behavior."""

    @pytest.fixture
    def text_file_ingestion_step(self):
        """Instantiate a `TextFileIngestionStep` with a mocked directory."""
        data_directory = "/mocked/path"
        file_extensions: List[TextFileExtension] = [".txt"]
        return TextFileIngestionStep(data_directory, file_extensions)

    @pytest.fixture
    def mocked_files(self):
        """Return a list of mocked files in the target directory."""
        return [
            Path("/mocked/path/file1.txt"),
            Path("/mocked/path/file2.txt"),
            Path("/mocked/path/file3.docx"),
        ]

    def test_process_with_valid_files(self, text_file_ingestion_step, mocked_files):
        """Process matching text files and yield `PipelineData` with proper metadata."""
        mock_file_contents = {
            "/mocked/path/file1.txt": "This is the content of file1.",
            "/mocked/path/file2.txt": "This is the content of file2.",
        }

        with (
            patch("builtins.open", mock_open()) as mocked_open,
            patch("pathlib.Path.iterdir", return_value=mocked_files),
        ):
            mocked_open.side_effect = lambda file, *args, **kwargs: mock_open(
                read_data=mock_file_contents[str(file)]
            ).return_value

            result = text_file_ingestion_step.process()

        assert len(result) == 2
        assert isinstance(result[0], PipelineData)
        assert result[0].text == "This is the content of file1."
        assert result[0].metadata["file_path"] == "/mocked/path/file1.txt"
        assert result[0].metadata["file_name"] == "file1.txt"

    def test_process_with_empty_directory(self, text_file_ingestion_step):
        """Return an empty list when the directory contains no files."""
        with patch("pathlib.Path.iterdir", return_value=[]):
            result = text_file_ingestion_step.process()

        assert result == []


class TestPdfFileIngestionStep:
    """Tests covering PDF file ingestion behavior."""

    @pytest.fixture
    def pdf_file_path(self, tmp_path):
        """Return a temporary PDF file path for ingestion tests."""
        return Path("/Users/user/Projects/AI_Osrodek/data/pdfs/Warszawska_1846.pdf")

    @pytest.fixture
    def pdf_ingestion_step_text(self, pdf_file_path):
        """Instantiate a `PdfFileIngestionStep` for text mode."""
        file_extensions: List[PDFFileExtension] = [".pdf"]
        return PdfFileIngestionStep(
            file_path=pdf_file_path, modes={"text"}, file_extensions=file_extensions
        )

    @pytest.fixture
    def pdf_ingestion_step_image(self, pdf_file_path):
        """Instantiate a `PdfFileIngestionStep` for image mode."""
        file_extensions: List[PDFFileExtension] = [".pdf"]
        return PdfFileIngestionStep(
            file_path=pdf_file_path,
            modes={"image"},
            page_range=(1, 10),
            file_extensions=file_extensions,
        )

    @pytest.fixture
    def pdf_ingestion_step_text_and_image(self, pdf_file_path):
        """Instantiate a `PdfFileIngestionStep` for text and image mode."""
        file_extensions: List[PDFFileExtension] = [".pdf"]
        return PdfFileIngestionStep(
            file_path=pdf_file_path,
            modes=("text", "image"),
            file_extensions=file_extensions,
        )

    def test_process_text_mode_with_pages(self, pdf_ingestion_step_text):
        """Process a PDF"""

        sample = next(iter(pdf_ingestion_step_text.iter_source()))

        assert sample is not None
        assert isinstance(sample, PipelineData)
        assert sample.text is not None

    def test_process_image_mode_with_pages(self, pdf_ingestion_step_image):

        sample = next(iter(pdf_ingestion_step_image.iter_source()))

        assert sample is not None
        assert isinstance(sample, PipelineData)
        assert sample.image is not None

    def test_process_text_and_image_mode_with_pages(
        self, pdf_ingestion_step_text_and_image
    ):

        sample = next(iter(pdf_ingestion_step_text_and_image.iter_source()))

        assert sample is not None
        assert isinstance(sample, PipelineData)
        assert sample.text is not None
        assert sample.image is not None
