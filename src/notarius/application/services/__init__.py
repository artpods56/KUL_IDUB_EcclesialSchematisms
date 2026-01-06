"""Application services - domain services and strategies."""

from notarius.application.services.data.merging import MergingService
from notarius.application.services.context.provider import (
    ComposedContextProvider,
    EmptyContextProvider,
    PageContentContextProvider,
    PreviousPageDomainContextProvider,
    PredictionsContextProvider,
    GroundTruthContextProvider,
)
from notarius.application.services.context.strategy import (
    AccumulatingStrategy,
    SlidingWindowStrategy,
    StatelessStrategy,
)
from notarius.application.services.processors.dataset_processor import (
    DatasetProcessor,
)
from notarius.application.services.processors.item_processor import (
    ItemProcessor,
    ItemProcessingResult,
    TextOnlyRequestHandler,
    TextExtractionResponseHandler,
)

from notarius.application.services.sequence_state import SequenceState

__all__ = [
    # Merging
    "MergingService",
    # Context providers
    "ComposedContextProvider",
    "EmptyContextProvider",
    "PageContentContextProvider",
    "PreviousPageDomainContextProvider",
    "PredictionsContextProvider",
    "GroundTruthContextProvider",
    # Conversation strategies
    "AccumulatingStrategy",
    "SlidingWindowStrategy",
    "StatelessStrategy",
    # Processors
    "DatasetProcessor",
    "ItemProcessor",
    "ItemProcessingResult",
    "TextOnlyRequestHandler",
    "TextExtractionResponseHandler",
    "SequenceState",
    "get_context_strategy",
    "ContextStrategySelection",
]

from notarius.application.services.context.strategy import (
    get_context_strategy,
    ContextStrategySelection,
)
