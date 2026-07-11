from notarius_core.application.context.provider import (
    ComposedContextProvider,
    PageContentContextProvider,
    PredictionsContextProvider,
    PreviousPageDomainContextProvider,
)
from notarius_schematisms.domain.models import SchematismPage

SCHEMATISM_REFINEMENT_CONTEXT_PROVIDERS = (
    PageContentContextProvider(),
    PageContentContextProvider(offset=1),
    PredictionsContextProvider(),
    PreviousPageDomainContextProvider(),
)

SOURCE_GENERATION_CONTEXT_PROVIDERS = (
    PageContentContextProvider(),
    PageContentContextProvider(offset=1),
    PreviousPageDomainContextProvider(),
)

TASK_SCHEMA_REGISTRY = {
    "structured_extraction": SchematismPage,
    "source_generation": SchematismPage,
    "elenchus_extraction": SchematismPage,
    "tr_1529_structured_extraction": SchematismPage,
    "transliterate_structured_extraction": SchematismPage,
    "xlm": SchematismPage,
}


def schematism_refinement_context_provider() -> ComposedContextProvider:
    return ComposedContextProvider(SCHEMATISM_REFINEMENT_CONTEXT_PROVIDERS)


def source_generation_context_provider() -> ComposedContextProvider:
    return ComposedContextProvider(SOURCE_GENERATION_CONTEXT_PROVIDERS)

