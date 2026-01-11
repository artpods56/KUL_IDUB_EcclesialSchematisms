"""Use case for generating source (Latin) dataset from parsed (Polish) ground truth."""

import asyncio
from dataclasses import dataclass
from itertools import chain
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
    AsyncBaseUseCase,
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
    AsyncBaseUseCase[GenerateSourceDatasetRequest, GenerateSourceDatasetResponse]
):
    """
    Use case for generating source (Latin) dataset from parsed (Polish) ground truth.

    This use case takes a merged dataset with parsed ground truth entries (Polish, normalized)
    and OCR text, then uses an LLM to find and extract the corresponding Latin text from
    page images. The result is a source dataset with Latin source entries.

    Processes schematism groups in parallel while keeping item processing
    sequential within each group (required for context strategy).
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
    async def execute(
        self, request: GenerateSourceDatasetRequest
    ) -> GenerateSourceDatasetResponse:
        """Execute the source generation workflow.

        Args:
            request: Request containing dataset and parameters

        Returns:
            Response with generated source dataset
        """
        if request.group_by_schematism_name:
            groups = list(request.dataset.group_by_schematism())
            total_groups = len(groups)

            logger.info(
                "Starting parallel source generation",
                total_schematisms=total_groups,
                schematism_names=[name for name, _ in groups],
            )

            async def process_group(
                index: int, schematism_name: str, dataset: PredictionItemDataset
            ) -> list[PredictionDataItem]:
                logger.info(
                    "Processing schematism",
                    schematism_name=schematism_name,
                    index=f"{index + 1}/{total_groups}",
                    items_count=len(dataset.items),
                )
                results = await asyncio.to_thread(
                    self.dataset_processor.process_sequence, dataset.items
                )
                return list(results)

            group_results = await asyncio.gather(
                *[process_group(i, name, dataset) for i, (name, dataset) in enumerate(groups)]
            )
            all_items = list(chain.from_iterable(group_results))
        else:
            results = await asyncio.to_thread(
                self.dataset_processor.process_sequence, request.dataset.items
            )
            all_items = list(results)

        return GenerateSourceDatasetResponse(
            dataset=PredictionItemDataset(items=all_items),
        )
