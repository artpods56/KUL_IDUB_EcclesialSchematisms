"""Conversation strategies for LLM dataset processing.

These strategies define how conversation history is managed across items
in a sequence. Each strategy controls what gets added to the conversation
and how state is updated.

Context Management for Sequential Source Processing
---------------------------------------------------
When processing ordered source items, each user message can include:
- CURRENT_ITEM_TEXT: OCR or source text of the current item
- NEXT_ITEM_TEXT: lookahead text for the next item
- PARSED_GROUND_TRUTH: Reference data to guide extraction
- PREVIOUS_ITEM_CONTEXT: inherited domain state

To avoid OCR duplication and manage context efficiently:
1. Strip images from history (reduce payload size)
2. Strip NEXT_PAGE_TEXT from historical user messages (prevent duplication)
3. Keep CURRENT_PAGE_TEXT in history (efficient "memory" of previous pages)
4. Keep all assistant responses (structured outputs with extracted context)

This allows the LLM to reference content from all previously processed pages
via their OCR text, without keeping expensive images or redundant text.
"""

import abc
from dataclasses import dataclass
from typing import Literal, Any, override

from PIL import Image

from notarius_core.prompts.message_builder import BaseMessageBuilder
from notarius_core.application.sequence_state import SequenceState
from notarius_core.domain.models.messages import (
    strip_images_from_message,
    strip_next_page_ocr_from_message,
)
from notarius_core.domain.models.conversation import Conversation


@dataclass
class BaseContextStrategy(abc.ABC):
    message_builder: BaseMessageBuilder
    """Template name to use for user messages."""

    @abc.abstractmethod
    def initialize_state(self) -> Conversation: ...

    """Called once before entering the sequence loop."""

    @abc.abstractmethod
    def prepare_state(
        self,
        state: SequenceState,
        context: dict[str, Any],
        image: Image.Image | None = None,
    ) -> SequenceState: ...

    """Called once before sending the request to the LLM."""

    @abc.abstractmethod
    def update_state(
        self,
        state: SequenceState,
    ) -> SequenceState: ...

    """Called once after receiving the response from the LLM."""


@dataclass
class StatelessStrategy(BaseContextStrategy):
    """No conversation history - each item is independent.

    Best for: OCR extraction, independent predictions where context
    from previous items doesn't matter.
    """

    @override
    def initialize_state(self) -> Conversation:
        """Initialize state with system message only."""
        return Conversation().add(
            self.message_builder.build_system_message(
                template_name="system.j2", context={}
            )
        )

    @override
    def prepare_state(
        self,
        state: SequenceState,
        context: dict[str, Any],
        image: Image.Image | None = None,
    ) -> SequenceState:
        """Prepare state by adding user message (stateless - no history)."""
        # Build user message using the message builder
        user_message = self.message_builder.build_user_message(
            template_name="user.j2", context=context, image=image
        )

        # For stateless strategy, start fresh with system + user message
        conversation = self.initialize_state().add(user_message)

        return SequenceState(
            conversation=conversation,
            domain_context=state.domain_context,
            items_processed=state.items_processed,
            current_item_index=state.current_item_index,
        )

    @override
    def update_state(
        self,
        state: SequenceState,
    ) -> SequenceState:
        """Update items_processed but don't accumulate history."""
        return SequenceState(
            conversation=Conversation(),
            domain_context=state.domain_context,
            items_processed=state.items_processed + 1,
            current_item_index=state.current_item_index,
        )


@dataclass
class AccumulatingStrategy(BaseContextStrategy):
    """Full conversation history - multi-turn with context.

    Best for: Sequential item processing where context carries over.

    Context management approach:
    - Keeps ALL assistant responses (structured outputs with context)
    - Keeps ALL user messages (with OCR text for reference)
    - Strips images from history (reduces payload size)
    - Strips NEXT_PAGE_TEXT from history (prevents OCR duplication)
    - Preserves CURRENT_PAGE_TEXT (efficient memory of previous pages)

    Why this works:
    - When processing item N+1, its current text contains what was lookahead
      text in item N. By stripping lookahead text from item N, we avoid
      duplication while preserving item N's source text.
    - The LLM can reference content from ALL previously processed items
      without keeping expensive images or redundant OCR text.
    - Context size grows linearly with number of pages, but efficiently
      (OCR text << base64 images).
    """

    strip_images: bool = True
    """Whether to strip images from history to reduce payload size."""

    @override
    def initialize_state(self) -> Conversation:
        """Initialize state with system message only."""
        return Conversation().add(
            self.message_builder.build_system_message("system.j2", {})
        )

    @override
    def prepare_state(
        self,
        state: SequenceState,
        context: dict[str, Any],
        image: Image.Image | None = None,
    ) -> SequenceState:

        user_message = self.message_builder.build_user_message(
            "user.j2", context, image
        )

        conversation = state.conversation.add(user_message)
        # conversation = self.initialize_state().add(user_message)

        return SequenceState(
            conversation=conversation,
            domain_context=state.domain_context,
            items_processed=state.items_processed,
            current_item_index=state.current_item_index,
        )

    @override
    def update_state(
        self,
        state: SequenceState,
    ) -> SequenceState:
        """Update state by stripping images and next-page OCR from history.

        Strips NEXT_PAGE_TEXT to prevent duplication while preserving
        CURRENT_PAGE_TEXT as efficient "memory" of each processed page.
        """
        conversation = state.conversation

        # Strip both images and NEXT_PAGE_TEXT sections to manage context efficiently
        stripped_messages = [
            strip_next_page_ocr_from_message(strip_images_from_message(msg))
            for msg in conversation.messages
        ]
        conversation = Conversation(messages=stripped_messages)

        return SequenceState(
            conversation=conversation,
            domain_context=state.domain_context,
            items_processed=state.items_processed + 1,
            current_item_index=state.current_item_index,
        )


@dataclass
class SlidingWindowStrategy(BaseContextStrategy):
    """Keep last N exchanges in conversation.

    Best for: Long sequences where full history is too expensive
    but some context is still useful.

    Context management approach:
    - Keeps system message (always)
    - Keeps last N exchanges (user + assistant pairs)
    - Strips older messages entirely (bounded context size)
    - Strips images from kept messages (reduces payload size)
    - Strips NEXT_PAGE_TEXT from kept messages (prevents OCR duplication)
    - Preserves CURRENT_PAGE_TEXT (efficient memory of recent pages)

    Why this works:
    - Bounded context size regardless of sequence length (cost-effective)
    - Recent items' source text remains available
    - Previous domain context carries forward essential state
    - No OCR duplication (NEXT_PAGE_TEXT stripped from history)
    - A summary field in domain context can preserve important information
      when older messages are dropped from the window

    Trade-offs:
    - Loses access to older pages' content beyond the window
    - Relies on previous domain context and summary fields for long-range context
    - More cost-effective than accumulating for very long sequences
    """

    window_size: int = 5
    """Number of exchanges (user + assistant pairs) to keep."""

    strip_images: bool = True
    """Whether to strip images from history."""

    @override
    def initialize_state(self) -> Conversation:
        """Initialize state with system message only."""
        return Conversation().add(
            self.message_builder.build_system_message(
                template_name="system.j2", context={}
            )
        )

    @override
    def prepare_state(
        self,
        state: SequenceState,
        context: dict[str, Any],
        image: Image.Image | None = None,
    ) -> SequenceState:
        """Prepare state by adding user message to windowed history."""
        # Build user message using the message builder
        user_message = self.message_builder.build_user_message(
            "user.j2", context, image
        )

        # Add to existing conversation history
        if state.conversation.messages:
            conversation = state.conversation.add(user_message)
        else:
            # First item - initialize with system message
            conversation = self.initialize_state().add(user_message)

        return SequenceState(
            conversation=conversation,
            domain_context=state.domain_context,
            items_processed=state.items_processed,
            current_item_index=state.current_item_index,
        )

    @override
    def update_state(
        self,
        state: SequenceState,
    ) -> SequenceState:
        """Apply sliding window and strip images and next-page OCR.

        Keeps only the last N exchanges while stripping NEXT_PAGE_TEXT
        to prevent duplication and CURRENT_PAGE_TEXT is preserved for
        efficient reference to recent pages.
        """
        # The conversation already has user + assistant messages
        # Domain context is already extracted by ResponseHandler
        conversation = state.conversation

        # Apply sliding window - keep system message + last N exchanges
        if conversation.messages:
            all_messages = list(conversation.messages)

            # Separate system message from exchanges
            system_messages = [msg for msg in all_messages if msg.role == "system"]
            exchange_messages = [msg for msg in all_messages if msg.role != "system"]

            # Keep only last N exchanges (N user+assistant pairs = 2N messages)
            max_exchange_messages = self.window_size * 2
            if len(exchange_messages) > max_exchange_messages:
                exchange_messages = exchange_messages[-max_exchange_messages:]

            # Reconstruct conversation
            windowed_messages = system_messages + exchange_messages
            conversation = Conversation(messages=windowed_messages)

        # Strip images and NEXT_PAGE_TEXT to manage context efficiently
        if conversation.messages:
            stripped_messages = [
                strip_next_page_ocr_from_message(strip_images_from_message(msg))
                for msg in conversation.messages
            ]
            conversation = Conversation(messages=stripped_messages)

        return SequenceState(
            conversation=conversation,
            domain_context=state.domain_context,
            items_processed=state.items_processed + 1,
            current_item_index=state.current_item_index,
        )


ContextStrategySelection = Literal["stateless", "accumulating", "sliding_window"]

CONTEXT_STRATEGY_MAPPING: dict[ContextStrategySelection, type[BaseContextStrategy]] = {
    "stateless": StatelessStrategy,
    "accumulating": AccumulatingStrategy,
    "sliding_window": SlidingWindowStrategy,
}


def get_context_strategy(
    strategy_literal: ContextStrategySelection,
    message_builder: BaseMessageBuilder,
) -> BaseContextStrategy:
    try:
        strategy_cls = CONTEXT_STRATEGY_MAPPING[strategy_literal]
    except KeyError:
        raise ValueError(
            f"Invalid context strategy: {strategy_literal}. ",
            f"Valid strategies: {CONTEXT_STRATEGY_MAPPING.keys()}",
        )

    return strategy_cls(message_builder=message_builder)
