"""Use case for enriching dataset with OCR predictions."""

from dataclasses import dataclass
from pathlib import Path
from typing import final, override, cast

from notarius.infrastructure.persistence.storage import ImageRepository
from structlog import get_logger

from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)
from notarius.infrastructure.cache.backends.ocr import create_ocr_cache_backend
from notarius.infrastructure.ocr import OCREngine, OCRRequest, OCRMode
from notarius.infrastructure.ocr.types import SimpleOCRResult
from notarius.schemas.data.pipeline import BaseDataset, BaseDataItem, BaseItemDataset
from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@dataclass
class EnrichWithOCRRequest(BaseRequest):
    """Request to enrich dataset with OCR predictions."""

    dataset: BaseDataset[BaseDataItem]
    mode: OCRMode = "text"


@dataclass
class EnrichWithOCRResponse(BaseResponse):
    """Response containing OCR-enriched dataset."""

    dataset: BaseDataset[BaseDataItem]
    ocr_executions: int
    cache_hits: int


@final
class EnrichDatasetWithOCR(BaseUseCase[EnrichWithOCRRequest, EnrichWithOCRResponse]):
    """
    Use case for enriching a dataset with OCR text predictions.

    This use case takes a dataset of items with images and enriches them with OCR text.
    Uses CachedEngine for automatic caching - no manual cache checking needed!
    """

    def __init__(
        self,
        ocr_engine: OCREngine,
        image_storage: ImageRepository,
        language: str = "lat+pol+rus",
        enable_cache: bool = True,
    ):
        """
        Initialize the use case.

        Args:
            ocr_engine: Engine for performing OCR predictions
            image_storage: Resource for loading images from storage
            language: OCR language configuration for cache namespacing
            enable_cache: Whether to enable caching (default: True)
        """
        self.image_storage = image_storage

        # Wrap engine with caching
        if enable_cache:
            backend, keygen = create_ocr_cache_backend(language)
            self.ocr_engine = CachedEngine(
                engine=ocr_engine,
                cache_backend=backend,
                key_generator=keygen,
                enabled=True,
            )
        else:
            self.ocr_engine = ocr_engine

    @override
    async def execute(self, request: EnrichWithOCRRequest) -> EnrichWithOCRResponse:
        """
        Execute the OCR enrichment workflow with automatic caching.

        Args:
            request: Request containing dataset and enrichment parameters

        Returns:
            Response with enriched dataset and execution statistics
        """
        dataset_len = len(request.dataset.items)
        new_dataset_items: list[BaseDataItem] = []

        for i, item in enumerate(request.dataset.items):
            if not item.image_path:
                logger.debug(f"Skipping item {i}/{dataset_len} - no image_path")
                continue

            image = self.image_storage.get(Path(item.image_path)).convert("RGB")
            logger.info(f"Processing {i + 1}/{dataset_len} sample with OCR.")

            # Process with cached engine - caching happens automatically!
            ocr_request = OCRRequest(input=image, mode=request.mode)
            response = self.ocr_engine.process(ocr_request)

            new_dataset_items.append(
                BaseDataItem(
                    image_path=item.image_path,
                    text=cast(SimpleOCRResult, response.output).text,
                    metadata=item.metadata,
                )
            )

        # Get stats from cached engine
        if isinstance(self.ocr_engine, CachedEngine):
            stats = self.ocr_engine.stats
            ocr_executions = stats["misses"]
            cache_hits = stats["hits"]
        else:
            ocr_executions = dataset_len
            cache_hits = 0

        logger.info(
            "OCR enrichment completed",
            total_items=dataset_len,
            ocr_executions=ocr_executions,
            cache_hits=cache_hits,
            cache_enabled=isinstance(self.ocr_engine, CachedEngine),
        )

        return EnrichWithOCRResponse(
            dataset=BaseItemDataset(items=new_dataset_items),
            ocr_executions=ocr_executions,
            cache_hits=cache_hits,
        )
