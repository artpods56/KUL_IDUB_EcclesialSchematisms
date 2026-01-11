"""Test factories for creating test data objects.

This package provides factory classes for creating real test objects instead
of using mocks. Factories create properly typed objects with sensible defaults,
improving test maintainability and type safety.

Philosophy:
- Use factories for domain entities, value objects, DTOs, and configs
- Factories create REAL objects, not mocks
- Each factory provides semantic methods for common scenarios
- Factories compose other factories for complex objects

Example:
    from tests.factories import SchematismPageFactory, BaseDatasetFactory

    # Create test data
    page = SchematismPageFactory.build(entry_count=5)
    dataset = BaseDatasetFactory.build(items=10)
"""

from tests.factories.base import BaseFactory
from tests.factories.entities import SchematismEntryFactory, SchematismPageFactory
from tests.factories.metadata import BaseMetaDataFactory, PageContextFactory
from tests.factories.datasets import (
    BaseDataItemFactory,
    BaseDatasetFactory,
    PredictionDataItemFactory,
    PredictionDatasetFactory,
    GroundTruthDataItemFactory,
    GroundTruthDatasetFactory,
    AlignedSchematismsDataItemFactory,
    AlignedDatasetFactory,
    EvaluationDataItemFactory,
    EvaluationDatasetFactory,
)
from tests.factories.messages import (
    TextContentFactory,
    ImageContentFactory,
    ChatMessageFactory,
    ConversationFactory,
)
from tests.factories.requests import (
    EnrichWithOCRRequestFactory,
    EnrichWithLMv3RequestFactory,
    IngestPDFRequestFactory,
    OCRRequestFactory,
    LMv3RequestFactory,
    CompletionRequestFactory,
)
from tests.factories.responses import (
    SimpleOCRResultFactory,
    StructuredOCRResultFactory,
    OCRResponseFactory,
    LMv3ResponseFactory,
    BaseProviderResponseFactory,
    CompletionResultFactory,
    EnrichWithOCRResponseFactory,
    EnrichWithLMv3ResponseFactory,
    IngestPDFResponseFactory,
)

__all__ = [
    # Base
    "BaseFactory",
    # Entities
    "SchematismEntryFactory",
    "SchematismPageFactory",
    # Metadata
    "BaseMetaDataFactory",
    "PageContextFactory",
    # Datasets
    "BaseDataItemFactory",
    "BaseDatasetFactory",
    "PredictionDataItemFactory",
    "PredictionDatasetFactory",
    "GroundTruthDataItemFactory",
    "GroundTruthDatasetFactory",
    "AlignedSchematismsDataItemFactory",
    "AlignedDatasetFactory",
    "EvaluationDataItemFactory",
    "EvaluationDatasetFactory",
    # Messages
    "TextContentFactory",
    "ImageContentFactory",
    "ChatMessageFactory",
    "ConversationFactory",
    # Requests
    "EnrichWithOCRRequestFactory",
    "EnrichWithLMv3RequestFactory",
    "IngestPDFRequestFactory",
    "OCRRequestFactory",
    "LMv3RequestFactory",
    "CompletionRequestFactory",
    # Responses
    "SimpleOCRResultFactory",
    "StructuredOCRResultFactory",
    "OCRResponseFactory",
    "LMv3ResponseFactory",
    "BaseProviderResponseFactory",
    "CompletionResultFactory",
    "EnrichWithOCRResponseFactory",
    "EnrichWithLMv3ResponseFactory",
    "IngestPDFResponseFactory",
]
