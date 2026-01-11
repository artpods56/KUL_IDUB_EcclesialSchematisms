"""Use case for enriching dataset with LayoutLMv3 predictions."""

from dataclasses import dataclass
from pathlib import Path
from typing import final, override, cast
from structlog import get_logger
from PIL import Image
from notarius.application.ports.outbound.cached_engine import CachedEngine
from notarius.application.ports.outbound.storage import AbstractFileRepository
from notarius.application.ports.outbound.engine import (
    ConfigurableEngine,
    CachedEngineStats,
)


from notarius.application.use_cases.use_case import (
    BaseRequest,
    BaseResponse,
    BaseUseCase,
)
from notarius.infrastructure.cache.backends.lmv3 import create_lmv3_cache_backend
from notarius.infrastructure.ml_models.lmv3.engine_adapter import (
    LMv3Request,
    LMv3Response,
)
from notarius.infrastructure.persistence.storage import ImageRepository
from notarius.schemas.configs import BaseLMv3ModelConfig
from notarius.schemas.data.pipeline import (
    BaseDataItem,
    BaseDataset,
    PredictionDataItem,
    PredictionItemDataset,
    BaseItemDataset,
)
from notarius.shared.logger import Logger

logger: Logger = get_logger(__name__)


@dataclass
class EnrichWithLMv3Request(BaseRequest):
    """Request to enrich dataset with LayoutLMv3 predictions."""

    dataset: BaseItemDataset


@dataclass
class EnrichWithLMv3Response(BaseResponse):
    """Response containing LayoutLMv3-enriched dataset."""

    dataset: PredictionItemDataset
    lmv3_executions: int
    cache_hits: int


@final
class EnrichDatasetWithLMv3(BaseUseCase[EnrichWithLMv3Request, EnrichWithLMv3Response]):
    """
    Use case for enriching a dataset with LayoutLMv3 predictions.

    This use case takes a dataset of items with images and enriches them with
    structured predictions from the LayoutLMv3 model. Uses CachedEngine for
    automatic caching - no manual cache checking needed!
    """

    def __init__(
        self,
        lmv3_engine: ConfigurableEngine[BaseLMv3ModelConfig, LMv3Request, LMv3Response],
        image_storage: AbstractFileRepository[Image.Image],
        checkpoint: str,
        enable_cache: bool = True,
    ):
        """
        Initialize the use case.

        Args:
            lmv3_engine: Engine for performing LayoutLMv3 predictions
            image_storage: Resource for loading images from storage
            checkpoint: Model checkpoint name for cache namespacing
            enable_cache: Whether to enable caching (default: True)
        """
        self.image_storage = image_storage

        # Wrap engine with caching
        if enable_cache:
            backend, keygen = create_lmv3_cache_backend(checkpoint)
            self.lmv3_engine = CachedEngine(
                engine=lmv3_engine,
                cache_backend=backend,
                key_generator=keygen,
                enabled=True,
            )
        else:
            self.lmv3_engine = lmv3_engine

    @override
    def execute(self, request: EnrichWithLMv3Request) -> EnrichWithLMv3Response:
        """
        Execute the LayoutLMv3 enrichment workflow with automatic caching.

        Args:
            request: Request containing dataset to enrich

        Returns:
            Response with enriched dataset and execution statistics
        """
        dataset_len = len(request.dataset.items)
        new_dataset_items: list[PredictionDataItem] = []

        for i, item in enumerate(request.dataset.items):
            if not item.image_path:
                logger.debug(f"Skipping item {i}/{dataset_len} - no image_path")
                continue

            image = self.image_storage.get(Path(item.image_path)).convert("RGB")
            logger.info(f"Processing {i + 1}/{dataset_len} sample with LMv3.")

            # Process with cached engine - caching happens automatically!
            lmv3_request = LMv3Request(input=image)
            response = self.lmv3_engine.process(lmv3_request)

            new_dataset_items.append(
                PredictionDataItem(
                    image_path=item.image_path,
                    text=item.text,
                    predictions=response.output,
                    metadata=item.metadata,
                )
            )

        # Get stats from cached engine
        if isinstance(self.lmv3_engine, CachedEngine):
            stats = cast(CachedEngineStats, self.lmv3_engine.stats)
            lmv3_executions = stats["misses"]
            cache_hits = stats["hits"]
        else:
            lmv3_executions = dataset_len
            cache_hits = 0

        logger.info(
            "LayoutLMv3 enrichment completed",
            total_items=dataset_len,
            lmv3_executions=lmv3_executions,
            cache_hits=cache_hits,
            cache_enabled=isinstance(self.lmv3_engine, CachedEngine),
        )

        return EnrichWithLMv3Response(
            dataset=PredictionItemDataset(items=new_dataset_items),
            lmv3_executions=lmv3_executions,
            cache_hits=cache_hits,
        )
