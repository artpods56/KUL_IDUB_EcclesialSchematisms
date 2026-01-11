"""Tests for JsonExportUseCase."""

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from notarius.application.use_cases.export.json_export import (
    JsonExportRequest,
    JsonExportResponse,
    JsonExportUseCase,
)
from tests.fakes.storage import FakeFileStorage
from notarius.schemas.data.pipeline import BaseDataItem, BaseDataset, BaseMetaData


@pytest.fixture
def fake_storage() -> FakeFileStorage:
    """Create a FakeFileStorage."""
    return FakeFileStorage(storage_root=Path("/storage/root"))


@pytest.fixture
def sample_metadata() -> BaseMetaData:
    """Create sample metadata."""
    return BaseMetaData(
        sample_id=1,
        schematism_name="test_schematism",
        filename="test_file.jpg",
    )


@pytest.fixture
def sample_dataset(sample_metadata: BaseMetaData) -> BaseDataset[BaseDataItem]:
    """Create a sample dataset with items."""
    items = [
        BaseDataItem(
            image_path="/path/to/image1.jpg",
            text="Sample text 1",
            metadata=sample_metadata,
        ),
        BaseDataItem(
            image_path="/path/to/image2.jpg",
            text="Sample text 2",
            metadata=BaseMetaData(
                sample_id=2,
                schematism_name="test_schematism",
                filename="test_file2.jpg",
            ),
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


@pytest.fixture
def multi_schematism_dataset() -> BaseDataset[BaseDataItem]:
    """Create a dataset with items from multiple schematisms."""
    items = [
        BaseDataItem(
            image_path="/path/to/image1.jpg",
            text="Text from schematism A",
            metadata=BaseMetaData(
                sample_id=1,
                schematism_name="schematism_a",
                filename="file1.jpg",
            ),
        ),
        BaseDataItem(
            image_path="/path/to/image2.jpg",
            text="Text from schematism B",
            metadata=BaseMetaData(
                sample_id=2,
                schematism_name="schematism_b",
                filename="file2.jpg",
            ),
        ),
        BaseDataItem(
            image_path="/path/to/image3.jpg",
            text="Another text from schematism A",
            metadata=BaseMetaData(
                sample_id=3,
                schematism_name="schematism_a",
                filename="file3.jpg",
            ),
        ),
    ]
    return BaseDataset[BaseDataItem](items=items)


@pytest.fixture
def empty_dataset() -> BaseDataset[BaseDataItem]:
    """Create an empty dataset."""
    return BaseDataset[BaseDataItem](items=[])


class TestJsonExportRequest:
    """Test suite for JsonExportRequest dataclass."""

    def test_request_with_defaults(
        self, sample_dataset: BaseDataset[BaseDataItem]
    ) -> None:
        """Test request creation with default values."""
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
        )

        assert request.dataset is sample_dataset
        assert request.output_dir == Path("/output")
        assert request.group_by_schematism is True
        assert request.pretty_print is True
        assert request.filename_prefix == ""

    def test_request_with_custom_values(
        self, sample_dataset: BaseDataset[BaseDataItem]
    ) -> None:
        """Test request creation with custom values."""
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/custom/output"),
            group_by_schematism=False,
            pretty_print=False,
            filename_prefix="custom_prefix",
        )

        assert request.output_dir == Path("/custom/output")
        assert request.group_by_schematism is False
        assert request.pretty_print is False
        assert request.filename_prefix == "custom_prefix"


class TestJsonExportResponse:
    """Test suite for JsonExportResponse dataclass."""

    def test_response_creation(self) -> None:
        """Test response creation with output files."""
        output_files = {
            "schematism_a": Path("/output/file_a.json"),
            "schematism_b": Path("/output/file_b.json"),
        }
        response = JsonExportResponse(output_files=output_files)

        assert response.output_files == output_files
        assert len(response.output_files) == 2


class TestJsonExportUseCase:
    """Test suite for JsonExportUseCase."""

    def test_init(self, fake_storage: FakeFileStorage) -> None:
        """Test initialization."""
        use_case = JsonExportUseCase(storage=fake_storage)
        assert use_case.storage is fake_storage

    def test_execute_with_single_schematism_grouped(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute with single schematism, grouped by schematism."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
            group_by_schematism=True,
        )

        response = use_case.execute(request)

        assert len(response.output_files) == 1
        assert "test_schematism" in response.output_files
        assert len(fake_storage.save_calls) == 1

        saved_content_bytes, saved_path = fake_storage.save_calls[0]
        assert saved_path.parent == Path("/output")

        saved_content = json.loads(saved_content_bytes.decode("utf-8"))
        assert saved_content["total_records"] == 2
        assert saved_content["schematism_name"] == "test_schematism"
        assert "generated_at" in saved_content
        assert len(saved_content["records"]) == 2

    def test_execute_without_grouping(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute without grouping by schematism."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
            group_by_schematism=False,
        )

        response = use_case.execute(request)

        assert len(response.output_files) == 1
        assert "all" in response.output_files
        assert len(fake_storage.save_calls) == 1

        saved_content_bytes, saved_path = fake_storage.save_calls[0]
        saved_content = json.loads(saved_content_bytes.decode("utf-8"))
        assert saved_content["total_records"] == 2
        assert "schematism_name" not in saved_content

    def test_execute_with_multiple_schematisms(
        self,
        fake_storage: FakeFileStorage,
        multi_schematism_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute with multiple schematisms grouped."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=multi_schematism_dataset,
            output_dir=Path("/output"),
            group_by_schematism=True,
        )

        response = use_case.execute(request)

        assert len(response.output_files) == 2
        assert "schematism_a" in response.output_files
        assert "schematism_b" in response.output_files
        assert len(fake_storage.save_calls) == 2

        all_calls = fake_storage.save_calls
        for saved_content_bytes, saved_path in all_calls:
            saved_content = json.loads(saved_content_bytes.decode("utf-8"))
            assert "schematism_name" in saved_content

        schematism_a_calls = [
            c
            for c in all_calls
            if "schematism_a" in json.loads(c[0].decode("utf-8"))["schematism_name"]
        ]
        assert len(schematism_a_calls) == 1
        schematism_a_content = json.loads(schematism_a_calls[0][0].decode("utf-8"))
        assert schematism_a_content["total_records"] == 2

    def test_execute_with_empty_dataset(
        self,
        fake_storage: FakeFileStorage,
        empty_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute with empty dataset."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=empty_dataset,
            output_dir=Path("/output"),
            group_by_schematism=False,
        )

        response = use_case.execute(request)

        assert len(response.output_files) == 1
        assert len(fake_storage.save_calls) == 1

        saved_content_bytes = fake_storage.save_calls[0][0]
        saved_content = json.loads(saved_content_bytes.decode("utf-8"))
        assert saved_content["total_records"] == 0
        assert len(saved_content["records"]) == 0

    def test_execute_with_filename_prefix(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute with custom filename prefix."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
            filename_prefix="export",
        )

        use_case.execute(request)

        saved_path = fake_storage.save_calls[0][1]
        assert "export_" in saved_path.name

    def test_execute_without_filename_prefix(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute without filename prefix."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
            filename_prefix="",
        )

        use_case.execute(request)

        saved_path = fake_storage.save_calls[0][1]
        filename_parts = saved_path.name.split("_")
        assert filename_parts[0].isdigit()

    def test_execute_with_pretty_print(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute with pretty print enabled."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
            pretty_print=True,
        )

        use_case.execute(request)

        saved_content_bytes = fake_storage.save_calls[0][0]
        json_str = saved_content_bytes.decode("utf-8")
        assert "\n" in json_str
        assert "  " in json_str

    def test_execute_without_pretty_print(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test execute with pretty print disabled."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
            pretty_print=False,
        )

        use_case.execute(request)

        saved_content_bytes = fake_storage.save_calls[0][0]
        json_str = saved_content_bytes.decode("utf-8")
        parsed = json.loads(json_str)
        minified = json.dumps(parsed, ensure_ascii=False)
        assert json_str == minified

    def test_execute_serializes_all_fields(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that all item fields are serialized correctly."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
        )

        use_case.execute(request)

        saved_content_bytes = fake_storage.save_calls[0][0]
        saved_content = json.loads(saved_content_bytes.decode("utf-8"))

        record = saved_content["records"][0]
        assert "image_path" in record
        assert "text" in record
        assert "metadata" in record
        assert record["image_path"] == "/path/to/image1.jpg"
        assert record["text"] == "Sample text 1"

    def test_execute_uses_storage_root_in_response(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that response includes storage_root in file paths."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("output"),
        )

        response = use_case.execute(request)

        for file_path in response.output_files.values():
            assert str(file_path).startswith("/storage/root")

    def test_execute_handles_unicode_content(
        self,
        fake_storage: FakeFileStorage,
    ) -> None:
        """Test that unicode content is handled correctly."""
        dataset = BaseDataset[BaseDataItem](
            items=[
                BaseDataItem(
                    image_path="/path/to/image.jpg",
                    text="Text with unicode: \u0142\u00f3d\u017a \u0141\u00d3D\u0179",
                    metadata=BaseMetaData(
                        sample_id=1,
                        schematism_name="unicode_test",
                        filename="test.jpg",
                    ),
                ),
            ]
        )

        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=dataset,
            output_dir=Path("/output"),
        )

        use_case.execute(request)

        saved_content_bytes = fake_storage.save_calls[0][0]
        json_str = saved_content_bytes.decode("utf-8")
        saved_content = json.loads(json_str)
        assert (
            saved_content["records"][0]["text"]
            == "Text with unicode: \u0142\u00f3d\u017a \u0141\u00d3D\u0179"
        )

    def test_execute_filename_format(
        self,
        fake_storage: FakeFileStorage,
        sample_dataset: BaseDataset[BaseDataItem],
    ) -> None:
        """Test that filenames follow expected format."""
        use_case = JsonExportUseCase(storage=fake_storage)
        request = JsonExportRequest(
            dataset=sample_dataset,
            output_dir=Path("/output"),
            filename_prefix="export",
        )

        use_case.execute(request)

        saved_path = fake_storage.save_calls[0][1]
        filename = saved_path.name

        assert filename.endswith(".json")
        assert "export_" in filename
        assert "test_schematism" in filename
        parts = filename.split("_")
        assert len(parts[0]) == 8
        assert parts[0].isdigit()
        assert len(parts[1]) == 6
        assert parts[1].isdigit()
        timestamp_full = f"{parts[0]}_{parts[1]}"
        assert len(timestamp_full) == 15
