from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, override, final, TypeVar, Generic

from notarius_core.ports.protocols import ContextProvider
from notarius_core.application.sequence_state import SequenceState
from notarius_core.domain.models.dataset import (
    BaseDataItem,
    PredictionDataItem,
    GroundTruthDataItem,
)


ItemT = TypeVar("ItemT", bound=BaseDataItem)


@dataclass
class PageContentContextProvider(Generic[ItemT], ContextProvider[ItemT]):
    offset: int = 0

    @property
    def position(self) -> str:
        if self.offset > 0:
            return "NEXT"
        elif self.offset < 0:
            return "PREVIOUS"
        else:
            return "CURRENT"

    @override
    def get_context(
        self,
        items: Sequence[ItemT],
        sequence_state: SequenceState,
    ) -> dict[str, Any]:
        """Provide page OCR text content with offset support.

        Always returns the key (e.g., NEXT_PAGE__TEXT), even when the page
        doesn't exist (e.g., last page has no next page). This ensures templates
        can safely check variables with StrictUndefined enabled.
        """
        try:
            key = f"{self.position}_PAGE__TEXT"
            return {key: items[sequence_state.current_item_index + self.offset].text}
        except IndexError:
            key = f"{self.position}_PAGE__TEXT"
            return {key: None}

    @override
    def get_context_keys(self) -> list[str]:
        return [f"{self.position}_PAGE__TEXT"]


@dataclass
class PreviousPageDomainContextProvider(ContextProvider[PredictionDataItem]):
    @override
    def get_context(
        self,
        items: Sequence[PredictionDataItem],
        sequence_state: SequenceState,
    ) -> dict[str, Any]:
        """Provide previous page context for sequential processing.

        Always returns the PREVIOUS_PAGE__CONTEXT key, even for the first page
        (where the value is None). This ensures templates can safely check the
        variable with StrictUndefined enabled.
        """
        try:
            if sequence_state.current_item_index <= 0:
                return {"PREVIOUS_PAGE__CONTEXT": None}

            previous_context = items[
                sequence_state.current_item_index - 1
            ].predictions.context
            return {
                "PREVIOUS_PAGE__CONTEXT": (
                    previous_context.model_dump() if previous_context else {}
                )
            }
        except (IndexError, AttributeError):
            return {"PREVIOUS_PAGE__CONTEXT": None}

    @override
    def get_context_keys(self) -> list[str]:
        return ["PREVIOUS_PAGE__CONTEXT"]


@dataclass
class PredictionsContextProvider(ContextProvider[PredictionDataItem]):
    @override
    def get_context(
        self,
        items: Sequence[PredictionDataItem],
        sequence_state: SequenceState,
    ) -> dict[str, Any]:
        try:
            return {
                "CURRENT_PAGE__PREDICTIONS": items[
                    sequence_state.current_item_index
                ].predictions.model_dump()
            }
        except (IndexError, AttributeError):
            return {"CURRENT_PAGE__PREDICTIONS": None}

    @override
    def get_context_keys(self) -> list[str]:
        return ["CURRENT_PAGE__PREDICTIONS"]


@dataclass
class GroundTruthContextProvider(ContextProvider[GroundTruthDataItem]):
    @override
    def get_context(
        self,
        items: Sequence[GroundTruthDataItem],
        sequence_state: SequenceState,
    ) -> dict[str, Any]:
        """Extract parsed entries from ground truth."""
        try:
            return {
                "CURRENT_PAGE__GROUND_TRUTH": items[
                    sequence_state.current_item_index
                ].model_dump()
            }
        except IndexError:
            return {"CURRENT_PAGE__GROUND_TRUTH": None}

    @override
    def get_context_keys(self) -> list[str]:
        return ["CURRENT_PAGE__GROUND_TRUTH"]


@final
@dataclass
class ComposedContextProvider(Generic[ItemT], ContextProvider[ItemT]):
    """Composes multiple context providers into one.

    Each provider's context is merged. Later providers override earlier ones
    if they have conflicting keys.
    """

    providers: Sequence[ContextProvider[ItemT]] = field(default_factory=list)

    @override
    def get_context(
        self,
        items: Sequence[ItemT],
        sequence_state: SequenceState,
    ) -> dict[str, Any]:
        """Gather context from all providers."""
        context: dict[str, Any] = {}
        for provider in self.providers:
            provider_context = provider.get_context(items, sequence_state)
            context.update(provider_context)
        return context

    @override
    def get_context_keys(self) -> list[str]:
        keys = []
        for provider in self.providers:
            keys.extend(provider.get_context_keys())
        return list(set(keys))


@final
@dataclass
class EmptyContextProvider(Generic[ItemT], ContextProvider[ItemT]):
    """Context provider that returns empty context.

    Best for: OCR extraction where no additional context is needed
    beyond the image and prompt template.
    """

    @override
    def get_context(
        self,
        items: Sequence[ItemT],
        sequence_state: SequenceState,
    ) -> dict[str, Any]:
        """Return empty context."""
        return {}

    @override
    def get_context_keys(self) -> list[str]:
        return []
