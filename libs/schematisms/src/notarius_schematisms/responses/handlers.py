from notarius_core.application.processors.item_processor import (
    StructuredOutputResponseHandler,
)
from notarius_schematisms.domain.dataset import PredictionDataItem
from notarius_schematisms.domain.models import SchematismPage


class SchematismResponseHandler(
    StructuredOutputResponseHandler[PredictionDataItem, SchematismPage]
):
    pass

