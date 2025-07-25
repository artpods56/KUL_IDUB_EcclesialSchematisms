from pathlib import Path
from typing import Any, Mapping

from jinja2 import (
    Environment,
    FileSystemLoader,
    select_autoescape,
    StrictUndefined,
)

from structlog import get_logger


logger = get_logger(__name__)


class PromptManager:
    """Lightweight wrapper around a `jinja2.Environment` for rendering LLM prompts.

    Improvements over the previous implementation:
    1) Uses `pathlib` for clearer path handling.
    2) Disables HTML auto-escaping for ``*.j2`` templates (prompts are plain text).
    3) Enables `StrictUndefined` so missing variables raise immediately.
    4) Removes superfluous whitespace with `trim_blocks` and `lstrip_blocks`.
    5) Adds basic debug logging through *structlog*.
    """

    def __init__(self, template_dir: str | Path = "prompts") -> None:
        base_dir = Path(template_dir)

        if not base_dir.is_absolute():
            # Resolve relative to the current file (src/core/models/llm/)
            base_dir = Path(__file__).resolve().parent / base_dir

        if not base_dir.exists():
            raise FileNotFoundError(f"Template directory '{base_dir}' does not exist")

        self.env = Environment(
            loader=FileSystemLoader(str(base_dir)),
            # Disable auto-escaping for .j2 (plain-text) templates
            autoescape=select_autoescape(disabled_extensions=("j2",), default=False, default_for_string=False),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        logger.debug("Prompt environment initialised", template_dir=str(base_dir))

    def render_prompt(self, template_name: str, **context: Mapping[str, Any]) -> str:
        """Render *template_name* with *context*.

        The method will raise ``jinja2.exceptions.UndefinedError`` if the
        template references a variable that is not provided in *context*.
        """
        template = self.env.get_template(template_name)
        rendered = template.render(**context)

        # Log at DEBUG level to avoid flooding standard INFO logs
        logger.debug(
            "Rendered prompt",
            template=template_name,
            context_keys=list(context.keys()),
            rendered_preview=rendered[:120] + ("…" if len(rendered) > 120 else ""),
        )

        return rendered
