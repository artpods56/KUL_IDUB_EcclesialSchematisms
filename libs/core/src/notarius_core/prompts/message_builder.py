"""Message builders for LLM dataset processing.

Provides implementations for building user messages from context
and optional images.
"""

import abc
from dataclasses import dataclass
from typing import Any, override

from PIL import Image

from notarius_core.domain.models.messages import ChatMessage
from notarius_core.ports.prompts import Jinja2PromptRenderer
from notarius_core.prompts.utils import (
    construct_image_message,
    construct_text_message,
)


@dataclass(frozen=True, kw_only=True)
class BaseMessageBuilder(abc.ABC):
    """Base class for message builders.

    Message builders are responsible for:
    - Rendering text prompts from context data
    - Constructing multimodal messages with images
    - Applying formatting and structure to user inputs
    """

    task_name: str
    """The task name used for template path construction."""

    def construct_template_name(self, template_name: str) -> str:
        """Construct full template path: tasks/{task_name}/{template_name}."""
        return f"tasks/{self.task_name}/{template_name}"

    @abc.abstractmethod
    def build_system_message(
        self, template_name: str, context: dict[str, Any]
    ) -> ChatMessage:
        """Build a system message from template and context."""
        ...

    @abc.abstractmethod
    def build_user_message(
        self,
        template_name: str,
        context: dict[str, Any],
        image: Image.Image | None,
    ) -> ChatMessage:
        """Build a user message from template, context, and optional image."""
        ...


@dataclass(frozen=True)
class Jinja2MessageBuilder(BaseMessageBuilder):
    """Jinja2-based message builder for rendering text prompts.

    Uses Jinja2 templates to render context data into text prompts,
    then constructs multimodal messages with optional images.
    """

    prompt_renderer: Jinja2PromptRenderer

    @override
    def build_system_message(
        self, template_name: str, context: dict[str, Any]
    ) -> ChatMessage:
        """Build system message from template and context."""
        text = self.prompt_renderer.render_prompt(
            self.construct_template_name(template_name), context
        )
        return construct_text_message(text=text, role="system")

    @override
    def build_user_message(
        self,
        template_name: str,
        context: dict[str, Any],
        image: Image.Image | None,
    ) -> ChatMessage:
        """Build user message from template, context and optional image.

        Args:
            template_name: Template name to use (e.g., "user.j2")
            context: Context data for template rendering
            image: Optional PIL Image to include in message

        Returns:
            ChatMessage with rendered text and optional image
        """
        text = self.prompt_renderer.render_prompt(
            self.construct_template_name(template_name), context
        )

        if image is not None:
            return construct_image_message(
                pil_image=image,
                text=text,
                role="user",
            )
        else:
            return construct_text_message(
                text=text,
                role="user",
            )
