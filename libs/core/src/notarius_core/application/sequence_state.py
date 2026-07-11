"""Immutable state carried while processing ordered extraction items."""

from dataclasses import dataclass
from typing import Any, Self

from pydantic import BaseModel

from notarius_core.domain.models.conversation import Conversation


@dataclass(frozen=True)
class SequenceState:
    """Conversation and contextual state shared between sequential items."""

    conversation: Conversation
    domain_context: BaseModel | dict[str, Any] | None = None
    items_processed: int = 0
    current_item_index: int = 0

    @classmethod
    def empty(cls) -> Self:
        return cls(
            conversation=Conversation(),
            domain_context=None,
            items_processed=0,
            current_item_index=0,
        )

