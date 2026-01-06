import io
import os
import pathlib
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO, Self, final, override, Literal, cast, Any

from PIL import Image

import notarius.domain.protocols
from notarius.application import ports
from notarius.domain import exceptions
from notarius import config


@final
class LocalFileStorage(ports.FileStorage):
    def _construct_full_path(self, file_path: pathlib.Path) -> pathlib.Path:
        return self.storage_root / file_path

    def __init__(self, storage_root: pathlib.Path) -> None:
        self.storage_root = storage_root

    @classmethod
    def from_config(cls, app_config: config.AppConfig) -> Self:
        return cls(storage_root=app_config.storage_root)

    @override
    def save(
        self,
        stream: notarius.domain.protocols.FileStreamProtocol,
        file_path: pathlib.Path,
    ) -> pathlib.Path:
        full_file_path = self._construct_full_path(file_path)

        try:
            with full_file_path.open("wb") as dest:
                shutil.copyfileobj(stream, dest)
            return full_file_path
        except OSError as e:
            raise exceptions.FileUploadError(
                f"Failed to save file {full_file_path} to the file system: {e}"
            ) from e

    @override
    @contextmanager
    def load(
        self, file_path: pathlib.Path
    ) -> Iterator[notarius.domain.protocols.FileStreamProtocol]:
        full_file_path = self._construct_full_path(file_path)

        try:
            file = io.open(full_file_path, "rb")
        except FileNotFoundError as e:
            raise exceptions.FileUploadError(
                f"Could not find file {full_file_path}: {e}"
            )
        except OSError as e:
            raise exceptions.FileDownloadError(
                f"Failed to download file {full_file_path} from the file system: {e}"
            ) from e

        try:
            yield file
        finally:
            file.close()

    @override
    def delete(self, file_path: pathlib.Path) -> None:
        full_file_path = self._construct_full_path(file_path)

        try:
            os.remove(full_file_path)
        except OSError as e:
            raise exceptions.FileDownloadError(
                f"Failed to remove file {full_file_path}: {e}"
            )

    @override
    def exists(self, file_path: pathlib.Path) -> bool:
        full_file_path = self._construct_full_path(file_path)
        return full_file_path.exists()


@final
class ImageRepository(ports.AbstractFileRepository[Image.Image]):

    def __init__(
        self,
        storage: ports.FileStorage,
        format: str = "JPEG",
        suffix: str = ".jpeg",
        save_kwargs: dict[str, Any] | None = None,
    ):
        self.storage = storage
        self.format = format
        self.suffix = suffix
        self.save_kwargs = save_kwargs or {}

    @override
    def add(self, file: Image.Image, name: str) -> pathlib.Path:
        buffer = io.BytesIO()
        image_to_save = file if file.mode == "RGB" else file.convert("RGB")
        try:
            image_to_save.save(buffer, format=self.format, **self.save_kwargs)
            buffer.seek(0)
            return self.storage.save(
                buffer, pathlib.Path(name).with_suffix(self.suffix)
            )
        finally:
            if image_to_save is not file:
                image_to_save.close()
            buffer.close()

    @override
    def get(self, path: pathlib.Path) -> Image.Image:
        with self.storage.load(path) as stream:
            image = Image.open(cast(BinaryIO, stream))
            image.load()
            if image.mode == "RGB":
                return image
            converted = image.convert("RGB")
            image.close()
            return converted

    @override
    def exists(self, name: str) -> bool:
        file_path = pathlib.Path(name).with_suffix(self.suffix)
        return self.storage.exists(file_path)

    @override
    def get_path(self, name: str) -> pathlib.Path:
        file_path = pathlib.Path(name).with_suffix(self.suffix)
        return self.storage.storage_root / file_path
