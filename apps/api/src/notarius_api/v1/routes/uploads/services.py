import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from PIL import Image as ImageModule
from PIL import ImageDraw

from notarius_api.services.errors import WorkbenchOperationError


_SAMPLE_PAGE_TEXTS = (
    "PAGE {index}\nParochia Sancti Floriani\nAnno Domini 1846",
    "PAGE {index}\nBaptisatorum liber\nVilla Nova, folio {index}",
    "PAGE {index}\nIndex nominum\nSeries continua",
)


@dataclass(frozen=True, slots=True)
class ImageUploadItem:
    upload_key: str
    filename: str
    byte_size: int


class ImageUploadService:
    """Stages opaque file uploads consumed by file-source nodes."""

    def __init__(self, uploads_dir: Path) -> None:
        self._uploads_dir = uploads_dir.expanduser().resolve()
        self._uploads_dir.mkdir(parents=True, exist_ok=True)

    def save_upload(
        self,
        filename: str,
        stream: BinaryIO,
    ) -> ImageUploadItem:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename).strip("-") or "upload"
        path = self._uploads_dir / f"{uuid4().hex[:8]}-{safe_name}"
        try:
            with path.open("xb") as destination:
                while chunk := stream.read(1024 * 1024):
                    destination.write(chunk)
        except OSError as exc:
            path.unlink(missing_ok=True)
            raise WorkbenchOperationError(
                f"Failed to stage upload {filename!r} in {self._uploads_dir}"
            ) from exc
        return ImageUploadItem(
            upload_key=path.name,
            filename=filename,
            byte_size=path.stat().st_size,
        )

    async def create_sample_images(
        self,
        count: int,
    ) -> list[ImageUploadItem]:
        items: list[ImageUploadItem] = []
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
                ImageUploadItem(
                    upload_key=path.name,
                    filename=f"sample-page-{index + 1}.png",
                    byte_size=path.stat().st_size,
                )
            )
        return items


__all__ = ["ImageUploadItem", "ImageUploadService"]
