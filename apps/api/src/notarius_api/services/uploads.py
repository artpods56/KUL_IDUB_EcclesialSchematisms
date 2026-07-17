import base64
import binascii
import re
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image as ImageModule
from PIL import ImageDraw

from notarius_api.schemas.workbench import ImageUploadItemResponse
from notarius_api.services.errors import WorkbenchOperationError


_SAMPLE_PAGE_TEXTS = (
    "PAGE {index}\nParochia Sancti Floriani\nAnno Domini 1846",
    "PAGE {index}\nBaptisatorum liber\nVilla Nova, folio {index}",
    "PAGE {index}\nIndex nominum\nSeries continua",
)


class ImageUploadService:
    """Stages image uploads consumed by image source nodes."""

    def __init__(self, uploads_dir: Path) -> None:
        self._uploads_dir = uploads_dir.expanduser().resolve()
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    async def save_image_upload(
        self,
        filename: str,
        content_base64: str,
    ) -> ImageUploadItemResponse:
        try:
            content = base64.b64decode(content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WorkbenchOperationError("Upload is not valid base64") from exc

        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-") or "upload"
        path = self._uploads_dir / f"{uuid4().hex[:8]}-{safe_name}"
        path.write_bytes(content)
        return ImageUploadItemResponse(
            upload_key=path.name,
            filename=filename,
            byte_size=path.stat().st_size,
        )

    async def create_sample_images(
        self,
        count: int,
    ) -> list[ImageUploadItemResponse]:
        items: list[ImageUploadItemResponse] = []
        for index in range(count):
            text = _SAMPLE_PAGE_TEXTS[index % len(_SAMPLE_PAGE_TEXTS)].format(
                index=index + 1
            )
            image = ImageModule.new("RGB", (420, 300), color="#f5f0e6")
            draw = ImageDraw.Draw(image)
            draw.rectangle((12, 12, 407, 287), outline="#b9ad98")
            draw.multiline_text((36, 48), text, fill="#463c2e", spacing=14)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            path = self._uploads_dir / f"{uuid4().hex[:8]}-sample-page.png"
            path.write_bytes(buffer.getvalue())
            items.append(
                ImageUploadItemResponse(
                    upload_key=path.name,
                    filename=f"sample-page-{index + 1}.png",
                    byte_size=path.stat().st_size,
                )
            )
        return items


__all__ = ["ImageUploadService"]
