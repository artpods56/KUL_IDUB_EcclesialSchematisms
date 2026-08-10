from typing import Annotated

from fastapi import Depends, Request

from notarius_core.application.modules import ModuleLibraryService

from notarius_api.app_state import get_resources


def module_library_service(request: Request) -> ModuleLibraryService:
    return get_resources(request.app).module_library


ModuleLibraryDependency = Annotated[
    ModuleLibraryService,
    Depends(module_library_service),
]


__all__ = ["ModuleLibraryDependency", "module_library_service"]
