"""Sequence state for LLM dataset processing.

Immutable state carried between items in a sequence, capturing
conversation history and domain-specific context.
"""

from dataclasses import dataclass
from typing import Self

from notarius.domain.entities.schematism import PageContext
from notarius.infrastructure.llm.conversation import Conversation


@dataclass(frozen=True)
class SequenceState:
    """Immutable state carried between items in a sequence.

    Captures everything needed for context accumulation:
    - Conversation history (for multi-turn)
    - Domain-specific context (e.g., PageContext for schematisms)
    - Processing metadata
    """

    conversation: Conversation
    """Accumulated conversation (previous user/assistant exchanges, not including system)."""

    domain_context: PageContext | None = None
    """Domain-specific context extracted from previous responses."""

    items_processed: int = 0
    """Number of items processed so far."""

    current_item_index: int = 0

    @classmethod
    def empty(cls) -> Self:
        return cls(
            conversation=Conversation(),
            domain_context=None,
            items_processed=0,
            current_item_index=0,
        )
