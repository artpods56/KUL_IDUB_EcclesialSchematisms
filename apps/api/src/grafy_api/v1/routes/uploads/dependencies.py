from typing import Annotated

from fastapi import Depends, Request

from grafy_api.app_state import get_resources

from .services import ImageUploadService


def image_upload_service(request: Request) -> ImageUploadService:
    return get_resources(request.app).uploads


ImageUploadDependency = Annotated[
    ImageUploadService,
    Depends(image_upload_service),
]


__all__ = ["ImageUploadDependency", "image_upload_service"]
