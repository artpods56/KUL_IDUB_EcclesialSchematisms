from typing import Any, Dict, Literal

from PIL.Image import Image
from pydantic import BaseModel, Field

from core.schemas.data.metrics import PageDataMetrics
from core.schemas.data.schematism import SchematismPage

PageDataSourceField = Literal[
    "ground_truth",
    "source_ground_truth",
    "lmv3_prediction",
    "llm_prediction",
    "parsed_prediction",
]


class PipelineData(BaseModel):
    """
    Pipeline data with required fields for ingestion and optional fields for pipeline stages
    """

    # Required fields - available at ingestion time
    image: Image | None = Field(default=None, description="Image used for prediction.")
    ground_truth: SchematismPage | None = Field(
        default=None, description="Ground truth data."
    )
    source_ground_truth: SchematismPage | None = Field(
        default=None, description="Source truth data."
    )

    # Optional fields - populated during pipeline processing
    text: str | None = Field(default=None, description="OCR text extracted from image.")
    language: str | None = Field(default=None, description="Detected language code.")
    language_confidence: float | None = Field(
        default=None, description="Confidence score for language detection."
    )

    # Model prediction fields
    lmv3_prediction: SchematismPage | None = Field(
        default=None, description="Predictions from LayoutLMv3 model."
    )
    llm_prediction: SchematismPage | None = Field(
        default=None, description="Predictions from LLM model."
    )
    parsed_prediction: SchematismPage | None = Field(
        default=None, description="Parsed predictions_data from LLM model."
    )
    parsed_messages: str | None = Field(
        default=None, description="Parsed messages from LLM model."
    )

    # Additional metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata for this pipeline."
    )
    evaluation_results: PageDataMetrics | None = Field(
        default=None, description="Results from evaluation step."
    )

    class Config:
        arbitrary_types_allowed = True
