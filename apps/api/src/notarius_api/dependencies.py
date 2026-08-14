from typing import Annotated

from fastapi import Depends, Request
from notarius_api.app_state import AppResources, get_resources
from notarius_api.settings import get_settings, Settings

def resources_dependency(request: Request) -> AppResources:
    return get_resources(request.app)

def settings_dependency(request: Request) -> Settings:
    return get_settings()

SettingsDependency = Annotated[Settings, Depends(settings_dependency)]

ResourcesDependency = Annotated[AppResources, Depends(resources_dependency)]