from typing import Annotated

from fastapi import Depends, Request

from grafy_core.application.templates import TemplateService

from grafy_api.app_state import get_resources


def template_service(request: Request) -> TemplateService:
    return get_resources(request.app).templates


TemplateDependency = Annotated[TemplateService, Depends(template_service)]


__all__ = ["TemplateDependency", "template_service"]
