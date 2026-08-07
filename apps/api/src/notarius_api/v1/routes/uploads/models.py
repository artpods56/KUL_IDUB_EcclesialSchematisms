from typing import Self

from pydantic import BaseModel, Field

from notarius_api.v1.models import ApiResponse

from .services import ImageUploadItem


class SampleRequest(BaseModel):
    count: int = Field(default=2, ge=1, le=8)


class ImageUploadItemResponse(ApiResponse):
    upload_key: str
    filename: str
    byte_size: int

    @classmethod
    def from_item(cls, item: ImageUploadItem) -> Self:
        return cls(
            upload_key=item.upload_key,
            filename=item.filename,
            byte_size=item.byte_size,
        )


__all__ = ["ImageUploadItemResponse", "SampleRequest"]
