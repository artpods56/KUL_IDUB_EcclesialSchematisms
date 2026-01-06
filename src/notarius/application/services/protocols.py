"""Protocols for LLM dataset processing components.

These protocols define the contracts for:
- Context extraction (what data goes into prompts)
- Conversation management (how history is handled)
- Result transformation (LLM output to dataset items)
"""

from collections.abc import Sequence
from typing import Protocol, Any, TypeVar, Final, TYPE_CHECKING

from PIL import Image

from notarius.schemas.data.pipeline import BaseDataItem

from notarius.application.services.sequence_state import SequenceState

from notarius.infrastructure.llm.conversation import Conversation
from notarius.domain.entities.messages import ChatMessage

if TYPE_CHECKING:
    from notarius.application.services.message_builder import BaseMessageBuilder

ItemT = TypeVar("ItemT", bound=BaseDataItem, contravariant=True)


class ContextProvider(Protocol[ItemT]):
    """Provides a piece of context for prompt rendering.

    Each provider contributes one aspect of context (OCR, hints, previous state).
    Providers are composed, not merged via dict update.
    """

    def get_context(
        self,
        items: Sequence[ItemT],
        sequence_state: SequenceState,
    ) -> dict[str, Any]:
        """Extract context relevant to this provider.

        Args:
            items: Current item being processed
            sequence_state: Optional state from previous items in sequence

        Returns:
            Dictionary of context key-value pairs
        """
        ...

    def get_context_keys(self) -> list[str]:
        """Return a list of keys that this provider contributes to the context.

        Returns:
            List of context keys
        """
        ...


class MessageBuilder(Protocol):
    """Builds user messages from context and optional image.

    Message builders are responsible for:
    - Rendering text prompts from context data
    - Constructing multimodal messages with images
    - Applying formatting and structure to user inputs
    """

    @property
    def task_name(self) -> str: ...

    def construct_template_name(self, template_name: str) -> str: ...

    def build_system_message(
        self, template_name: str, context: dict[str, Any]
    ) -> ChatMessage: ...

    def build_user_message(
        self,
        template_name: str,
        context: dict[str, Any],
        image: Image.Image | None,
    ) -> ChatMessage: ...


class ContextStrategy(Protocol):
    """Defines how conversation history is managed across items.

    Strategies control:
    - Whether to add previous exchanges to conversation
    - What to strip (images) before adding
    - How many turns to keep (sliding window)
    """

    @property
    def message_builder(self) -> "BaseMessageBuilder":
        """The message builder for constructing user messages."""
        ...

    def initialize_state(self) -> Conversation: ...

    def prepare_state(
        self,
        state: SequenceState,
        context: dict[str, Any],
        image: Image.Image | None = None,
    ) -> SequenceState: ...

    def update_state(self, state: SequenceState) -> SequenceState: ...


class ConversationOrchestrator(Protocol):
    """Orchestrates conversation assembly from components.

    Coordinates message building with conversation history management
    to create complete conversations for LLM processing.
    """

    def prepare_conversation(
        self,
        user_message: ChatMessage,
        base_conversation: Conversation,
        sequence_state: SequenceState,
    ) -> Conversation: ...
