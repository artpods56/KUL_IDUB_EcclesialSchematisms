import logging
import os

from jinja2 import Environment, FileSystemLoader, select_autoescape

from logger.setup import setup_logging

# setup_logging()
# logger = logging.getLogger(__name__)


class PromptManager:
    def __init__(self, template_dir="prompts"):

        if not os.path.isabs(template_dir):
            template_dir = os.path.join(os.path.dirname(__file__), template_dir)

        assert os.path.exists(
            template_dir
        ), f"Template directory '{template_dir}' does not exist"

        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def render_prompt(self, template_name, **context):
        template = self.env.get_template(template_name)
        return template.render(context)


prompt_manager = PromptManager()
