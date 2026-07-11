"""Sequential and parallel dataset processing orchestration."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from structlog import get_logger

from notarius_core.application.processors.item_processor import ItemProcessor
from notarius_core.application.sequence_state import SequenceState
from notarius_core.domain.models.dataset import BaseDataItem
from notarius_core.ports.protocols import ContextProvider, ContextStrategy
from notarius_core.ports.storage import ImageRepositoryProtocol

logger = get_logger(__name__)


class DatasetProcessor[ItemT: BaseDataItem, OutputT: BaseModel]:
    def __init__(
        self,
        item_processor: ItemProcessor[ItemT, OutputT],
        images_repository: ImageRepositoryProtocol,
        context_provider: ContextProvider[ItemT],
        context_strategy: ContextStrategy,
    ):
        self.item_processor = item_processor
        self.image_repository = images_repository
        self.context_provider = context_provider
        self.context_strategy = context_strategy

    def process_sequence(self, items: Sequence[ItemT]) -> Sequence[ItemT]:
        sequence_state = SequenceState.empty()

        for index, item in enumerate(items):
            current_state = SequenceState(
                conversation=sequence_state.conversation,
                domain_context=sequence_state.domain_context,
                items_processed=sequence_state.items_processed,
                current_item_index=index,
            )

            if not item.image_path:
                logger.debug("Skipping item without image_path", index=index)
                continue

            image = self.image_repository.get(Path(item.image_path)).convert("RGB")
            context = self.context_provider.get_context(items, current_state)
            prepared_state = self.context_strategy.prepare_state(
                image=image,
                context=context,
                state=current_state,
            )
            processing_result = self.item_processor.process(
                item=item,
                state=prepared_state,
            )
            sequence_state = self.context_strategy.update_state(
                state=processing_result.state
            )

        return items

    async def process_parallel_async(
        self,
        items: Sequence[ItemT],
        max_concurrent: int = 10,
    ) -> Sequence[ItemT]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(index: int, item: ItemT) -> None:
            async with semaphore:
                if not item.image_path:
                    logger.debug("Skipping item without image_path", index=index)
                    return
                current_state = SequenceState(
                    conversation=self.context_strategy.initialize_state(),
                    domain_context=None,
                    items_processed=0,
                    current_item_index=index,
                )
                image = await asyncio.to_thread(
                    lambda: self.image_repository.get(Path(item.image_path)).convert(
                        "RGB"
                    )
                )
                context = self.context_provider.get_context(items, current_state)
                prepared_state = self.context_strategy.prepare_state(
                    state=current_state,
                    context=context,
                    image=image,
                )
                processing_result = await self.item_processor.process_async(
                    item=item,
                    state=prepared_state,
                )
                self.context_strategy.update_state(state=processing_result.state)

        await asyncio.gather(*[process_one(i, item) for i, item in enumerate(items)])
        return items

