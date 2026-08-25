from typing import cast

from grafy_core.runtime.resolvers import Resolver

from grafy_plugin_ocr import tesseract
from grafy_plugin_ocr.artifacts import OCR_PAGE_RESULT
from grafy_plugin_ocr.declaration import OCR
from grafy_plugin_ocr.persistence import OcrPageResultOutputWriter
from grafy_plugin_ocr.resolvers import PilImageResolver

_NODE_MODULES = (tesseract,)

OCR.register_artifact_type(OCR_PAGE_RESULT)

OCR.register_resolver(
    lambda context: cast(
        Resolver[object],
        PilImageResolver(uow=context.uow, storage=context.storage),
    )
)

OCR.register_writer(
    lambda context: OcrPageResultOutputWriter(uow=context.uow, engine="fake")
)


__all__ = ["OCR"]
