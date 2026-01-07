"""Use case for refining schematism predictions using LLM."""

from dataclasses import dataclass
from typing import final, override, Any

from pydantic import BaseModel

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
    BaseUseCase,
)
from notarius.schemas.data.pipeline import (
    PredictionDataItem,
    PredictionItemDataset,
)


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
    BaseUseCase[RefinePredictionsRequest, RefinePredictionsResponse]
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
    def execute(self, request: RefinePredictionsRequest) -> RefinePredictionsResponse:
        """Execute LLM prediction refinement workflow.

        Args:
            request: Request containing datasets and prediction parameters

        Returns:
            Response with predicted dataset and execution statistics
        """
        all_items: list[PredictionDataItem] = []

        if request.group_by_schematism_name:
            for _, dataset in request.dataset.group_by_schematism():
                results = self.dataset_processor.process_sequence(items=dataset.items)
                for input_item, output in zip(dataset.items, results):
                    all_items.append(
                        PredictionDataItem(
                            image_path=input_item.image_path,
                            text=output.text,
                            metadata=input_item.metadata,
                            predictions=output.predictions,
                        )
                    )
        else:
            results = self.dataset_processor.process_sequence(
                items=request.dataset.items
            )
            for input_item, output in zip(request.dataset.items, results):
                all_items.append(
                    PredictionDataItem(
                        image_path=input_item.image_path,
                        text=output.text,
                        metadata=input_item.metadata,
                        predictions=output.predictions,
                    )
                )

        return RefinePredictionsResponse(
            dataset=PredictionItemDataset(items=all_items),
        )
