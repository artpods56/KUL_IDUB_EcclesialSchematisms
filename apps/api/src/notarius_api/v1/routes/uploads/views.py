from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from notarius_api.services.errors import WorkbenchOperationError

from .dependencies import ImageUploadDependency
from .models import ImageUploadItemResponse, SampleRequest


router = APIRouter(tags=["workbench"])


@router.post("/uploads", response_model=ImageUploadItemResponse)
async def upload_file(
    service: ImageUploadDependency,
    file: UploadFile = File(),
) -> ImageUploadItemResponse:
    if not file.filename:
        raise HTTPException(status_code=422, detail="Upload filename is required")
    try:
        item = await run_in_threadpool(
            service.save_upload,
            file.filename,
            file.file,
        )
        return ImageUploadItemResponse.from_item(item)
    except WorkbenchOperationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/samples", response_model=list[ImageUploadItemResponse])
async def create_samples(
    request: SampleRequest,
    service: ImageUploadDependency,
) -> list[ImageUploadItemResponse]:
    items = await service.create_sample_images(request.count)
    return [ImageUploadItemResponse.from_item(item) for item in items]


__all__ = ["router"]
