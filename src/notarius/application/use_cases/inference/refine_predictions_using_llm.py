"""Use case for refining schematism predictions using LLM."""

import asyncio
from dataclasses import dataclass
from itertools import chain
from typing import final, override, Any

from notarius.application.services import (
    PageContentContextProvider,
    PreviousPageDomainContextProvider,
    PredictionsContextProvider,
    ComposedContextProvider,
)
from notarius.application.services.processors.dataset_processor import DatasetProcessor
from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    AsyncBaseUseCase,
)
from notarius.schemas.data.pipeline import (
    PredictionDataItem,
    PredictionItemDataset,
)
from notarius.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RefinePredictionsRequest(BaseRequest):
    """Request to refine predictions with LLM model."""

    dataset: PredictionItemDataset
    group_by_schematism_name: bool = True


@dataclass
class RefinePredictionsResponse(BaseResponse):
    """Response containing refined LLM predictions."""

    dataset: PredictionItemDataset
    # execution_stats: dict[str, Any]


PREDICTIONS_REFINEMENT_CONTEXT_PROVIDERS = ComposedContextProvider(
    providers=[
        PageContentContextProvider[PredictionDataItem](offset=0),
        PageContentContextProvider[PredictionDataItem](offset=1),
        PreviousPageDomainContextProvider(),
        PredictionsContextProvider(),
    ]
)


@final
class RefinePredictionsWithLLM(
    AsyncBaseUseCase[RefinePredictionsRequest, RefinePredictionsResponse]
):

    def __init__(
        self,
        dataset_processor: DatasetProcessor[PredictionDataItem, Any],
    ):
        """Initialize use case.

        Args:
            dataset_processor: Processor for handling the sequence of items
        """
        self.dataset_processor = dataset_processor

    @override
    async def execute(
        self, request: RefinePredictionsRequest
    ) -> RefinePredictionsResponse:
        """Execute LLM prediction refinement workflow.

        Processes schematism groups in parallel while keeping item processing
        sequential within each group (required for context strategy).

        Args:
            request: Request containing datasets and prediction parameters

        Returns:
            Response with predicted dataset and execution statistics
        """
        if request.group_by_schematism_name:
            groups = list(request.dataset.group_by_schematism())
            total_groups = len(groups)

            logger.info(
                "Starting parallel refinement",
                total_schematisms=total_groups,
                schematism_names=[name for name, _ in groups],
            )

            async def process_group(
                index: int, name: str, dataset: PredictionItemDataset
            ) -> list[PredictionDataItem]:
                logger.info(
                    "Processing schematism",
                    schematism_name=name,
                    index=f"{index + 1}/{total_groups}",
                    items_count=len(dataset.items),
                )
                results = await asyncio.to_thread(
                    self.dataset_processor.process_sequence, dataset.items
                )
                return [
                    PredictionDataItem(
                        image_path=input_item.image_path,
                        text=output.text,
                        metadata=input_item.metadata,
                        predictions=output.predictions,
                    )
                    for input_item, output in zip(dataset.items, results)
                ]

            group_results = await asyncio.gather(
                *[
                    process_group(i, name, dataset)
                    for i, (name, dataset) in enumerate(groups)
                ]
            )
            all_items = list(chain.from_iterable(group_results))
        else:
            results = await asyncio.to_thread(
                self.dataset_processor.process_sequence, request.dataset.items
            )
            all_items = [
                PredictionDataItem(
                    image_path=input_item.image_path,
                    text=output.text,
                    metadata=input_item.metadata,
                    predictions=output.predictions,
                )
                for input_item, output in zip(request.dataset.items, results)
            ]

        return RefinePredictionsResponse(
            dataset=PredictionItemDataset(items=all_items),
        )
