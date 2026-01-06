import dataclasses
from dataclasses import dataclass, field

from notarius.domain.entities.messages import ChatMessage, ChatMessageList


@dataclass(frozen=True)
class Conversation:
    """Immutable input history using domain message types."""

    messages: ChatMessageList = field(default_factory=tuple)
    max_history_length: int | None = None

    def truncate_history(self, message_count: int) -> "Conversation":
        """Truncate history to the specified number of messages."""
        raise NotImplementedError

    def add(self, message: ChatMessage) -> "Conversation":
        """Add a single message to the input."""
        return Conversation(messages=(*self.messages, message))

    def add_many(self, messages: ChatMessageList) -> "Conversation":
        """Add multiple messages to the input."""
        return Conversation(messages=(*self.messages, *messages))

    def to_list(self) -> ChatMessageList:
        """Convert to a mutable message list for _engine consumption."""
        return list(*self.messages)

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return dataclasses.asdict(self)

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "Conversation":
        """Create Conversation from a list of messages.

        Args:
            messages: List of ChatMessage instances

        Returns:
            Conversation instance
        """
        return cls(messages=tuple(messages))

    def last_user_message(self) -> ChatMessage | None:
        """Get the last user message from the conversation.

        Returns:
            The last ChatMessage with role "user", or None if not found
        """
        for message in reversed(self.messages):
            if message.role == "user":
                return message
        return None
