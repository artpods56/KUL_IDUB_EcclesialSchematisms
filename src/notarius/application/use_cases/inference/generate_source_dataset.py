"""Use case for generating source (Latin) dataset from parsed (Polish) ground truth."""

from dataclasses import dataclass
from typing import Any, final, override

from notarius.application.services import (
    DatasetProcessor,
    ComposedContextProvider,
    PageContentContextProvider,
    PreviousPageDomainContextProvider,
    GroundTruthContextProvider,
)
from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)
from notarius.schemas.data.pipeline import (
    PredictionDataItem,
    PredictionItemDataset,
    BaseDataItem,
)
from notarius.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerateSourceDatasetRequest(BaseRequest):
    """Request to generate source (Latin) dataset from parsed (Polish) ground truth.

    The dataset should contain PredictionDataItem with:
    - predictions: The parsed Polish ground truth (converted from ground_truth)
    - text: OCR text for the page
    """

    dataset: PredictionItemDataset
    group_by_schematism_name: bool = True


@dataclass
class GenerateSourceDatasetResponse(BaseResponse):
    """Response containing generated source (Latin) dataset."""

    dataset: PredictionItemDataset


SOURCE_GENERATION_CONTEXT_PROVIDERS = ComposedContextProvider[Any](
    providers=[
        PageContentContextProvider[BaseDataItem](offset=0),
        PageContentContextProvider[BaseDataItem](offset=1),
        PreviousPageDomainContextProvider(),
        GroundTruthContextProvider(),
    ]
)


@final
class GenerateSourceDataset(
    BaseUseCase[GenerateSourceDatasetRequest, GenerateSourceDatasetResponse]
):
    """
    Use case for generating source (Latin) dataset from parsed (Polish) ground truth.

    This use case takes a merged dataset with parsed ground truth entries (Polish, normalized)
    and OCR text, then uses an LLM to find and extract the corresponding Latin text from
    page images. The result is a source dataset with Latin source entries.

    Uses DatasetProcessor with AccumulatingStrategy for sequential processing.
    """

    def __init__(
        self,
        dataset_processor: DatasetProcessor[PredictionDataItem, Any],
    ):
        """Initialize the use case.

        Args:
            dataset_processor: Processor for handling the sequence of items
        """
        self.dataset_processor = dataset_processor

    @override
    def execute(
        self, request: GenerateSourceDatasetRequest
    ) -> GenerateSourceDatasetResponse:
        """Execute the source generation workflow.

        Args:
            request: Request containing dataset and parameters

        Returns:
            Response with generated source dataset
        """
        all_items: list[PredictionDataItem] = []

        if request.group_by_schematism_name:
            for schematism_name, dataset in request.dataset.group_by_schematism():
                logger.info(
                    "Processing schematism",
                    schematism_name=schematism_name,
                    items_count=len(dataset.items),
                )
                results = self.dataset_processor.process_sequence(items=dataset.items)
                all_items.extend(results)
        else:
            results = self.dataset_processor.process_sequence(
                items=request.dataset.items
            )
            all_items.extend(results)

        return GenerateSourceDatasetResponse(
            dataset=PredictionItemDataset(items=all_items),
        )
