from io import BytesIO

import pymupdf

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    List,
    Literal,
    Iterator,
    Any,
)

from PIL import Image
from omegaconf import DictConfig
from tqdm import tqdm

from core.data.utils import get_dataset
from core.pipeline.steps.base import IngestionProcessingStep
from core.schemas.data.pipeline import PipelineData
from core.schemas.data.schematism import SchematismPage


ImageFileExtension = Literal[".jpg", ".jpeg", ".png"]
TextFileExtension = Literal[".txt"]
PDFFileExtension = Literal[".pdf"]


@dataclass
class FilterMapSpec:
    """Specification for a filter or map operation with its required columns.

    This allows clean specification of operations that need specific columns,
    avoiding the current issue where all filters/maps share the same input_columns.

    Args:
        operation: The filter/map function to apply.
        input_columns: Optional list of columns this operation needs.
            If None, all columns are passed to the operation.
        operation_type: Either 'filter' or 'map' to specify the operation type.

    Example:
        # Filter that needs only schematism_name column
        FilterMapSpec(
            operation=filter_schematisms(["krakow_1871", "warszawa_1872"]),
            input_columns=["schematism_name"],
            operation_type="filter"
        )

        # Map that needs multiple columns
        FilterMapSpec(
            operation=some_mapping_function,
            input_columns=["image", "labels"],
            operation_type="map"
        )
    """

    operations: list[Callable[[Any], Any]]
    input_columns: List[str] | None = None
    operation_type: Literal["filter", "map"] = "filter"
    negate: bool = False


class ImageFileIngestionStep(IngestionProcessingStep[PipelineData]):

    def __init__(self, data_directory: Path, file_extensions: List[ImageFileExtension]):
        super().__init__()
        self.data_directory = data_directory
        self.file_extensions = file_extensions

    def iter_source(self, *args: Any, **kwargs: Any) -> Iterator[PipelineData]:
        for file in sorted(self.data_directory.iterdir()):
            if file.suffix.lower() in self.file_extensions:
                ground_truth = None
                gt_path = file.with_suffix(".json")
                if gt_path.exists():
                    with open(gt_path, "r", encoding="utf-8") as f:
                        ground_truth = SchematismPage.model_validate_json(f.read())

                with Image.open(file) as image:
                    # Assumes filename file_format: {schematism}_{original_filename}
                    parts = file.stem.split("_", 1)
                    schematism_name = parts[0] if len(parts) > 1 else "unknown"
                    original_file_name = parts[1] if len(parts) > 1 else file.name

                    pipeline_data = PipelineData(
                        image=image.copy(),
                        ground_truth=ground_truth,
                        metadata={
                            "file_path": str(file),
                            "file_name": original_file_name,
                            "schematism_name": schematism_name,
                        },
                    )

                    yield pipeline_data


class TextFileIngestionStep(IngestionProcessingStep[PipelineData]):

    def __init__(self, data_directory: Path, file_extensions: List[TextFileExtension]):
        super().__init__()
        self.data_directory = data_directory
        self.file_extensions = file_extensions

    def iter_source(self) -> Iterator[PipelineData]:

        for file in self.data_directory.iterdir():
            if file.suffix.lower() in self.file_extensions:
                ground_truth = None
                gt_path = file.with_suffix(".json")
                if gt_path.exists():
                    with open(gt_path, "r", encoding="utf-8") as f:
                        ground_truth = SchematismPage.model_validate_json(f.read())

                with open(file, "r") as text_file:
                    text = text_file.read()

                    # Assumes filename file_format: {schematism}_{original_filename}
                    parts = file.stem.split("_", 1)
                    schematism_name = parts[0] if len(parts) > 1 else "unknown"
                    original_file_name = parts[1] if len(parts) > 1 else file.name

                    pipeline_data = PipelineData(
                        text=text,
                        ground_truth=ground_truth,
                        metadata={
                            "file_path": str(file),
                            "file_name": original_file_name,
                            "schematism_name": schematism_name,
                        },
                    )

                    yield pipeline_data


class HuggingFaceIngestionStep(IngestionProcessingStep[Any]):
    """HuggingFace dataset ingestion with improved filter/map API.

    This step loads a HuggingFace dataset and applies specified filters and maps.
    The new API allows each filter/map to specify its required columns independently.

    Args:
        dataset_config: Configuration for the HuggingFace dataset.
        filter_map_specs: List of FilterMapSpec objects defining operations to apply.
        wrapper: Whether to wrap the dataset in a DatasetWrapper.

    Example:
        step = HuggingFaceIngestionStep(
            dataset_config=config,
            filter_map_specs=[
                FilterMapSpec(
                    operation=filter_schematisms(["krakow_1871"]),
                    input_columns=["schematism_name"],
                    operation_type="filter"
                ),
                FilterMapSpec(
                    operation=some_map_function,
                    input_columns=["image", "labels"],
                    operation_type="map"
                )
            ]
        )
    """

    def __init__(
        self,
        dataset_config: DictConfig,
        filter_map_specs: List[FilterMapSpec] | None = None,
        yield_count: bool = False,
        wrapper: bool = False,
    ):
        super().__init__()

        # Load base dataset
        self.positive_samples = dataset_config.positive_samples
        self.negative_samples = dataset_config.negative_samples
        self.dataset = get_dataset(dataset_config, wrapper=wrapper)
        self.yield_count = yield_count
        self.ground_truth_column = dataset_config["column_map"]["ground_truth_column"]

        if filter_map_specs:
            self._apply_filter_map_specs(filter_map_specs)

    def _apply_filter_map_specs(self, specs: List[FilterMapSpec]) -> None:

        def make_filter(
            op: Callable[[dict[str, Any]], bool], negate: bool
        ) -> Callable[[dict[str, Any]], bool]:
            def _f(x: dict[str, Any]) -> bool:
                return not op(x) if negate else op(x)

            return _f

        """Apply filter and map specifications to the dataset."""
        for spec in specs:
            if spec.operation_type == "filter":
                for operation in spec.operations:
                    self.dataset = self.dataset.filter(
                        make_filter(operation, spec.negate),
                        input_columns=spec.input_columns,  # type: ignore
                    )
            elif spec.operation_type == "map":
                for operation in spec.operations:
                    self.dataset = self.dataset.map(
                        make_filter(operation, spec.negate),
                        input_columns=spec.input_columns,  # type: ignore
                    )
            else:
                raise ValueError(f"Unknown operation_type: {spec.operation_type}")

    def iter_source(self) -> Iterator[Any]:
        if not self.yield_count:
            yield from self.dataset
            return

        positive_needed = self.positive_samples
        negative_needed = self.negative_samples

        for sample in self.dataset:
            schematism_page_data = sample[self.ground_truth_column]
            page_data_entries_len = len(schematism_page_data["entries"])

            if page_data_entries_len > 0 and positive_needed > 0:
                positive_needed -= 1
                yield sample
            elif page_data_entries_len == 0 and negative_needed > 0:
                negative_needed -= 1
                yield sample

            if positive_needed <= 0 and negative_needed <= 0:
                break


class PdfFileIngestionStep(IngestionProcessingStep[PipelineData]):

    def __init__(
        self,
        file_path: Path,
        modes: set[Literal["image", "text"]],
        page_range: tuple[int, int] | None = None,
        file_extensions: PDFFileExtension = "pdf",
    ):
        super().__init__()
        self.file_path = file_path
        self.start_page, self.end_page = (
            page_range if page_range is not None else (None, None)
        )
        self.file_extensions = file_extensions
        self.modes = modes

    def iter_source(self, **kwargs: Any) -> Iterator[PipelineData]:

        with pymupdf.open(self.file_path) as pdf:

            if self.start_page and self.end_page:
                pages_range = pdf.pages(self.start_page, self.end_page)
            else:
                pages_range = pdf.pages()

            for page in pages_range:
                pipeline_data = {}

                pipeline_data["metadata"] = {"schematism": self.file_path.stem}

                if "text" in self.modes:
                    text = page.get_text()
                    if not text:
                        self.logger.warning(f"No text found in page: {page}")
                    pipeline_data["text"] = text

                if "image" in self.modes:

                    pix = page.get_pixmap()
                    image = pix.pil_image()
                    if not image:
                        self.logger.warning(f"No image found in page: {page}")

                    pipeline_data["image"] = image.convert("L").convert("RGB")

                if pipeline_data:
                    pipeline_data["metadata"]["pdf_page_number"] = page.number

                    yield PipelineData(**pipeline_data)
