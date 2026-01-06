"""Item processor for LLM dataset processing.

Generic processor for a single dataset item that handles LLM calls
with structured output.
"""

from dataclasses import dataclass
from typing import Never, final, override, Protocol, runtime_checkable

from pydantic import BaseModel
from structlog import get_logger

from notarius.application.services.sequence_state import SequenceState
from notarius.domain.entities.schematism import SchematismPage
from notarius.infrastructure.llm.engine_adapter import (
    CompletionRequest,
    CompletionResult,
    LLMCompletionEngine,
)
from notarius.schemas.data.pipeline import BaseDataItem, PredictionDataItem
from notarius.shared.logger import Logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ItemProcessingResult[ItemT: BaseDataItem, T]:
    state: SequenceState
    item: ItemT
    response: T


@runtime_checkable
class RequestHandler[T: BaseModel](Protocol):
    """Protocol for building LLM requests from sequence state."""

    def handle_request(self, state: SequenceState) -> CompletionRequest[T]: ...


@final
class StandardRequestHandler[T: BaseModel](RequestHandler[T]):
    """Default implementation that builds a simple completion request."""

    def __init__(self, output_type: type[T]):
        self.output_type = output_type

    @override
    def handle_request(self, state: SequenceState) -> CompletionRequest[T]:
        return CompletionRequest(
            input=state.conversation,
            structured_output=self.output_type,
        )


class ResponseHandler[ItemT: BaseDataItem, T: BaseModel](Protocol):
    """Protocol for handling LLM responses and updating item state."""

    def handle_response(
        self, item: ItemT, state: SequenceState, response: CompletionResult[T]
    ) -> ItemProcessingResult[ItemT, T]: ...


class PredictionsRefinementResponseHandler[ItemT: PredictionDataItem](
    ResponseHandler[ItemT, SchematismPage]
):
    """Generic response handler that just extracts structured output."""

    @override
    def handle_response(
        self,
        item: ItemT,
        state: SequenceState,
        response: CompletionResult[SchematismPage],
    ) -> ItemProcessingResult[ItemT, SchematismPage]:
        output = response.output.structured_response
        if output is None:
            raise ValueError("LLM returned no structured response")

        item.predictions = output

        updated_state = SequenceState(
            conversation=response.updated_conversation,
            domain_context=output.context,
            items_processed=state.items_processed,
            current_item_index=state.current_item_index,
        )

        return ItemProcessingResult(
            state=updated_state,
            item=item,
            response=output,
        )


@final
class TextOnlyRequestHandler(RequestHandler[Never]):
    """Request handler for text-only LLM calls without structured output.

    Best for: OCR extraction where we want raw text response.
    """

    @override
    def handle_request(self, state: SequenceState) -> CompletionRequest[Never]:
        return CompletionRequest(
            input=state.conversation,
            structured_output=None,
        )


@final
class TextExtractionResponseHandler[ItemT: BaseDataItem](ResponseHandler[ItemT, Never]):
    """Response handler that extracts text and sets item.text.

    Best for: OCR extraction where we want to populate the text field.
    """

    @override
    def handle_response(
        self,
        item: ItemT,
        state: SequenceState,
        response: CompletionResult[Never],
    ) -> ItemProcessingResult[ItemT, Never]:
        item.text = response.output.text_response

        updated_state = SequenceState(
            conversation=response.updated_conversation,
            domain_context=state.domain_context,
            items_processed=state.items_processed,
            current_item_index=state.current_item_index,
        )

        return ItemProcessingResult(  # pyright: ignore[reportReturnType]
            state=updated_state,
            item=item,
            response=None,
        )


@final
class ItemProcessor[ItemT: BaseDataItem, OutputT: BaseModel]:
    def __init__(
        self,
        llm_engine: LLMCompletionEngine,
        request_handler: RequestHandler[OutputT],
        response_handler: ResponseHandler[ItemT, OutputT],
    ):
        """Initialize the item processor.

        Args:
            llm_engine: Engine for making LLM calls
            request_handler: Handler for building requests
            response_handler: Handler for processing responses
        """
        self.llm_engine = llm_engine
        self.request_handler = request_handler
        self.response_handler = response_handler

    def process(
        self,
        item: ItemT,
        state: SequenceState,
    ) -> ItemProcessingResult[ItemT, OutputT]:
        """Process a single item with LLM.

        Args:
            item: Item to process
            state: Current sequence state

        Returns:
            Processing result with messages and structured response
        """
        request = self.request_handler.handle_request(state)

        logger.debug(
            "Processing item",
            item_id=item.metadata.sample_id if item.metadata else None,
            filename=item.metadata.filename if item.metadata else None,
        )

        result = self.llm_engine.process(request)

        return self.response_handler.handle_response(item, state, result)

    async def process_async(
        self,
        item: ItemT,
        state: SequenceState,
    ) -> ItemProcessingResult[ItemT, OutputT]:
        """Process a single item with LLM asynchronously.

        Args:
            item: Item to process
            state: Current sequence state

        Returns:
            Processing result with messages and structured response
        """
        request = self.request_handler.handle_request(state)

        logger.debug(
            "Processing item async",
            item_id=item.metadata.sample_id if item.metadata else None,
            filename=item.metadata.filename if item.metadata else None,
        )

        result = await self.llm_engine.process_async(request)

        return self.response_handler.handle_response(item, state, result)
