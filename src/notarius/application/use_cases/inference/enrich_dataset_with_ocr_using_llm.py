"""Use case for enriching dataset with OCR predictions using LLM via OpenRouter."""

from dataclasses import dataclass
from typing import Any, final, override


from notarius.application.services import (
    DatasetProcessor,
)
from notarius.application.use_cases.use_case import (
    AsyncBaseUseCase,
    BaseRequest,
    BaseResponse,
)
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem
from notarius.shared.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnrichWithLLMOCRRequest(BaseRequest):
    """Request to enrich dataset with LLM-based OCR predictions."""

    dataset: BaseDataset[BaseDataItem]
    group_by_schematism_name: bool = True
    max_concurrent_requests: int = 10


@dataclass
class EnrichWithLLMOCRResponse(BaseResponse):
    """Response containing LLM OCR-enriched dataset."""

    dataset: BaseDataset[BaseDataItem]


@final
class EnrichDatasetWithLLMOCR(
    AsyncBaseUseCase[EnrichWithLLMOCRRequest, EnrichWithLLMOCRResponse]
):
    """
    Use case for enriching a dataset with OCR text using LLM vision capabilities.

    Uses DatasetProcessor with StatelessStrategy for parallel async processing.
    """

    def __init__(
        self,
        dataset_processor: DatasetProcessor[BaseDataItem, Any],
    ):

        self.dataset_processor = dataset_processor

    @override
    async def execute(
        self, request: EnrichWithLLMOCRRequest
    ) -> EnrichWithLLMOCRResponse:
        """
        Execute the LLM OCR enrichment workflow with async concurrent processing.
        """
        if request.group_by_schematism_name:
            for schematism_name, dataset in request.dataset.group_by_schematism():
                logger.info(
                    "Processing schematism",
                    schematism_name=schematism_name,
                    items_count=len(dataset.items),
                )
                _ = await self.dataset_processor.process_parallel_async(
                    items=dataset.items,
                    max_concurrent=request.max_concurrent_requests,
                )
        else:
            _ = await self.dataset_processor.process_parallel_async(
                items=request.dataset.items,
                max_concurrent=request.max_concurrent_requests,
            )


        return EnrichWithLLMOCRResponse(
            dataset=request.dataset,
        )
