"""Dataset processor for LLM dataset processing.

Thin orchestrator that handles iteration over a sequence of items,
coordinating context gathering, message building, and LLM processing.
"""

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import final

from pydantic import BaseModel

from notarius.application.services.processors.item_processor import ItemProcessor
from notarius.application.services.protocols import (
    ContextProvider,
    ContextStrategy,
)
from notarius.application.services.sequence_state import SequenceState
from notarius.infrastructure.persistence.storage import ImageRepository
from notarius.schemas.data.pipeline import BaseDataItem
from notarius.shared.logger import get_logger

logger = get_logger(__name__)


class DatasetProcessor[ItemT: BaseDataItem, OutputT: BaseModel]:
    def __init__(
        self,
        item_processor: ItemProcessor[ItemT, OutputT],
        images_repository: ImageRepository,
        context_provider: ContextProvider[ItemT],
        context_strategy: ContextStrategy,
    ):
        self.item_processor = item_processor
        self.image_repository = images_repository
        self.context_provider = context_provider
        self.context_strategy = context_strategy

        logger.debug(
            f"ContexProvider supplies the following context keys: {self.context_provider.get_context_keys()}"
        )

    def process_sequence(self, items: Sequence[ItemT]) -> Sequence[ItemT]:
        total_items = len(items)
        logger.info("Starting sequence processing", total_items=total_items)

        sequence_state = SequenceState.empty()

        for i, item in enumerate(items):
            current_state = SequenceState(
                conversation=sequence_state.conversation,
                domain_context=sequence_state.domain_context,
                items_processed=sequence_state.items_processed,
                current_item_index=i,
            )

            if not item.image_path:
                logger.debug(f"Skipping item {i} - no image_path")
                continue

            logger.info(
                "Processing sequence item",
                index=i + 1,
                total=total_items,
                progress=f"{(i + 1) / total_items * 100:.1f}%",
            )

            image = self.image_repository.get(Path(item.image_path)).convert("RGB")

            context = self.context_provider.get_context(items, current_state)

            prepared_state = self.context_strategy.prepare_state(
                image=image, context=context, state=current_state
            )

            processing_result = self.item_processor.process(
                item=item, state=prepared_state
            )

            updated_state = self.context_strategy.update_state(
                state=processing_result.state
            )

            sequence_state = updated_state

        return items

    async def process_parallel_async(
        self,
        items: Sequence[ItemT],
        max_concurrent: int = 10,
    ) -> Sequence[ItemT]:
        """Process items in parallel with rate limiting.

        Each item is processed independently with fresh state (no accumulation).
        Best for stateless processing like OCR where items don't depend on each other.

        Args:
            items: Items to process
            max_concurrent: Maximum concurrent LLM requests

        Returns:
            Processed items with results populated
        """
        total_items = len(items)
        logger.info(
            "Starting parallel processing",
            total_items=total_items,
            max_concurrent=max_concurrent,
        )

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(i: int, item: ItemT) -> None:
            async with semaphore:
                if not item.image_path:
                    logger.debug(f"Skipping item {i} - no image_path")
                    return

                logger.info(
                    "Processing item in parallel",
                    index=i + 1,
                    total=total_items,
                )

                current_state = SequenceState(
                    conversation=self.context_strategy.initialize_state(),
                    domain_context=None,
                    items_processed=0,
                    current_item_index=i,
                )

                image_path = item.image_path
                image = await asyncio.to_thread(
                    lambda: self.image_repository.get(Path(image_path)).convert("RGB")
                )

                context = self.context_provider.get_context(items, current_state)

                prepared_state = self.context_strategy.prepare_state(
                    state=current_state, context=context, image=image
                )

                processing_result = await self.item_processor.process_async(
                    item=item, state=prepared_state
                )

                _ = self.context_strategy.update_state(state=processing_result.state)

        _ = await asyncio.gather(
            *[process_one(i, item) for i, item in enumerate(items)]
        )

        return items
