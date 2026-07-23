from typing import cast

from notarius_core.artifacts import Artifact
from notarius_core.runtime.persistence import InlineModelOutputWriter
from notarius_core.runtime.resolvers import InlineModelResolver, Resolver

from notarius_plugin_ocr import mistral, tables, tesseract
from notarius_plugin_ocr.artifacts import (
    MISTRAL_OCR_RESPONSE,
    OCR_PAGE_RESULT,
    TABLE_FRAGMENT,
)
from notarius_plugin_ocr.declaration import OCR
from notarius_plugin_ocr.mistral import MistralOcrResponsePayload
from notarius_plugin_ocr.persistence import OcrPageResultOutputWriter
from notarius_plugin_ocr.resolvers import (
    EncodedPageImageResolver,
    PilImageResolver,
)

_NODE_MODULES = (mistral, tables, tesseract)

OCR.register_artifact_type(OCR_PAGE_RESULT)

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
OCR.register(
    Artifact(
        spec=MISTRAL_OCR_RESPONSE,
        resolver=lambda context: InlineModelResolver(
            source=MISTRAL_OCR_RESPONSE.key,
            target=MistralOcrResponsePayload,
            uow=context.uow,
        ),
        writer=lambda context: InlineModelOutputWriter(
            artifact_type=MISTRAL_OCR_RESPONSE.key,
            model=MistralOcrResponsePayload,
            uow=context.uow,
        ),
    )
)

OCR.register_writer(
    lambda context: OcrPageResultOutputWriter(uow=context.uow, engine="fake")
)
OCR.register_artifact_type(TABLE_FRAGMENT)
OCR.register_writer(
    lambda context: InlineModelOutputWriter(
        artifact_type=TABLE_FRAGMENT.key,
        model=tables.TableFragment,
        uow=context.uow,
    )
)


__all__ = ["OCR"]
