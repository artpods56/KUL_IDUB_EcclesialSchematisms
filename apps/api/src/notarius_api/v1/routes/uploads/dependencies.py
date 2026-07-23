from typing import Annotated

from fastapi import Depends, Request

from .services import ImageUploadService


def image_upload_service(request: Request) -> ImageUploadService:
    service = getattr(request.app.state, "image_uploads", None)
    if not isinstance(service, ImageUploadService):
        raise RuntimeError("Image upload service is not initialized")
    return service


ImageUploadDependency = Annotated[
    ImageUploadService,
    Depends(image_upload_service),
]


__all__ = ["ImageUploadDependency", "image_upload_service"]
