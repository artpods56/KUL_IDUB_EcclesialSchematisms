from pathlib import Path

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    meta,
    select_autoescape,
)
from structlog import get_logger


logger = get_logger(__name__)


class Jinja2PromptRenderer:
    """Jinja2-based implementation of PromptRenderer.

    This adapter wraps a `jinja2.Environment` for rendering LLM prompts.

    Features:
    1) Uses `pathlib` for clearer path handling.
    2) Disables HTML auto-escaping for ``*.j2`` templates (prompts are plain text).
    3) Enables `StrictUndefined` so missing variables raise immediately.
    4) Removes superfluous whitespace with `trim_blocks` and `lstrip_blocks`.
    5) Includes are resolved relative to the template's folder first, then the root.
    """

    def __init__(self, template_dir: str | Path = "prompts") -> None:
        base_dir = Path(template_dir)

        if not base_dir.is_absolute():
            base_dir = Path.cwd() / base_dir

        if not base_dir.exists():
            raise FileNotFoundError(f"Template directory '{base_dir}' does not exist")

        self.base_dir = base_dir

    def _create_env(self, search_paths: list[str]) -> Environment:
        """Create a Jinja2 environment with the given search paths."""
        return Environment(
            loader=FileSystemLoader(search_paths),
            undefined=StrictUndefined,  # Raise error on undefined variables
            autoescape=select_autoescape(
                disabled_extensions=("j2",), default=False, default_for_string=False
            ),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def get_template_variables(self, template_name: str) -> set[str]:
        """Extract all variables referenced in a template (including includes).

        This parses the template and returns all variable names that the
        template expects to receive in its context. Useful for validation.

        Args:
            template_name: Path to template relative to base_dir

        Returns:
            Set of variable names used in the template

        Example:
            >>> renderer = Jinja2PromptRenderer()
            >>> variables = renderer.get_template_variables("tasks/source_generation/user.j2")
            >>> print(variables)
            {'CURRENT_PAGE__TEXT', 'NEXT_PAGE__TEXT', 'PREVIOUS_PAGE__CONTEXT', ...}
        """
        template_path = self.base_dir / template_name
        template_dir = template_path.parent
        search_paths = [str(template_dir), str(self.base_dir)]

        env = self._create_env(search_paths)

        # Get template source - loader is guaranteed to exist in FileSystemLoader
        if env.loader is None:
            raise RuntimeError("Template loader not initialized")
        template_source = env.loader.get_source(env, template_path.name)[0]

        # Parse and extract variables
        parsed = env.parse(template_source)
        return meta.find_undeclared_variables(parsed)

    def validate_context(
        self,
        template_name: str,
        context: dict[str, str],
        strict: bool = False
    ) -> dict[str, list[str]]:
        """Validate that context matches template variables.

        Args:
            template_name: Template to validate against
            context: Context dictionary to validate
            strict: If True, require exact match (no extra context keys)

        Returns:
            Dictionary with validation results:
            - 'missing': Variables used in template but not in context
            - 'unused': Variables in context but not used in template (only if strict=True)

        Example:
            >>> renderer = Jinja2PromptRenderer()
            >>> context = {'CURRENT_PAGE__TEXT': '...', 'TYPO_KEY': '...'}
            >>> result = renderer.validate_context('user.j2', context, strict=True)
            >>> if result['missing']:
            ...     print(f"Missing variables: {result['missing']}")
            >>> if result['unused']:
            ...     print(f"Unused variables: {result['unused']}")
        """
        template_vars = self.get_template_variables(template_name)
        context_keys = set(context.keys())

        missing = template_vars - context_keys
        unused = context_keys - template_vars if strict else set()

        if missing:
            logger.warning(
                "Template variables missing from context",
                template=template_name,
                missing=sorted(missing),
            )

        if unused and strict:
            logger.debug(
                "Context keys not used in template",
                template=template_name,
                unused=sorted(unused),
            )

        return {
            'missing': sorted(missing),
            'unused': sorted(unused),
        }

    def render_prompt(
        self,
        template_name: str,
        context: dict[str, str],
        validate: bool = False
    ) -> str:
        """Render *template_name* with *context*.

        Includes are resolved relative to the template's folder first,
        then fall back to the prompts root folder.

        The method will raise ``jinja2.exceptions.UndefinedError`` if the
        template references a variable that is not provided in *context*.

        Args:
            template_name: Path to template relative to base_dir
            context: Template variables
            validate: If True, validate context before rendering (logs warnings)

        Returns:
            Rendered prompt string

        Raises:
            jinja2.exceptions.UndefinedError: If template uses undefined variable
        """
        # Optional validation before rendering
        if validate:
            validation = self.validate_context(template_name, context, strict=False)
            if validation['missing']:
                # StrictUndefined will raise anyway, but this gives a clearer error
                missing_vars = ", ".join(validation['missing'])
                raise ValueError(
                    f"Template '{template_name}' requires variables that are not in context: {missing_vars}"
                )

        # Get the template's parent directory for relative includes
        template_path = self.base_dir / template_name
        template_dir = template_path.parent

        # Search paths: template's folder first, then root
        search_paths = [str(template_dir), str(self.base_dir)]

        env = self._create_env(search_paths)

        # Load template by filename only (since template_dir is in search path)
        template = env.get_template(template_path.name)
        rendered = template.render(**context)
        return rendered
