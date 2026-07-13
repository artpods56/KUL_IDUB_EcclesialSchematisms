from typing import cast

from notarius_core.runtime.persistence import InlineModelOutputWriter
from notarius_core.runtime.resolvers import InlineModelResolver, Resolver

from notarius_plugin_ocr import tables, tesseract
from notarius_plugin_ocr.artifacts import (
    MISTRAL_OCR_RESPONSE,
    OCR_PAGE_RESULT,
)
from notarius_plugin_ocr.declaration import OCR
from notarius_plugin_ocr.mistral import MistralOcrResponsePayload
from notarius_plugin_ocr.persistence import OcrPageResultOutputWriter
from notarius_plugin_ocr.resolvers import (
    EncodedPageImageResolver,
    PilImageResolver,
)

_NODE_MODULES = (tables, tesseract)

OCR.register_artifact_type(OCR_PAGE_RESULT)
OCR.register_artifact_type(MISTRAL_OCR_RESPONSE)

OCR.register_resolver(
    lambda context: cast(
        Resolver[object],
        PilImageResolver(uow=context.uow, storage=context.storage),
    )
)
OCR.register_resolver(
    lambda context: cast(
        Resolver[object],
        EncodedPageImageResolver(uow=context.uow, storage=context.storage),
    )
)
OCR.register_resolver(
    lambda context: cast(
        Resolver[object],
        InlineModelResolver(
            source=MISTRAL_OCR_RESPONSE.key,
            target=MistralOcrResponsePayload,
            uow=context.uow,
        ),
    )
)

OCR.register_writer(
    lambda context: OcrPageResultOutputWriter(uow=context.uow, engine="fake")
)
OCR.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=MISTRAL_OCR_RESPONSE.key,
        model=MistralOcrResponsePayload,
        uow=context.uow,
    )
)


__all__ = ["OCR"]
