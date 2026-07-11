"""Generic processor for one extraction item."""

from dataclasses import dataclass
from typing import Never, Protocol, final, override, runtime_checkable

from pydantic import BaseModel
from structlog import get_logger

from notarius_core.application.sequence_state import SequenceState
from notarius_core.domain.models.dataset import BaseDataItem, PredictionDataItem
from notarius_core.ports.llm import (
    CompletionRequest,
    CompletionResult,
    LLMCompletionEngine,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class ItemProcessingResult[ItemT: BaseDataItem, T]:
    state: SequenceState
    item: ItemT
    response: T


@runtime_checkable
class RequestHandler[T: BaseModel](Protocol):
    def handle_request(self, state: SequenceState) -> CompletionRequest[T]: ...


@final
class StandardRequestHandler[T: BaseModel](RequestHandler[T]):
    def __init__(self, output_type: type[T]):
        self.output_type = output_type

    @override
    def handle_request(self, state: SequenceState) -> CompletionRequest[T]:
        return CompletionRequest(
            input=state.conversation,
            structured_output=self.output_type,
        )


class ResponseHandler[ItemT: BaseDataItem, T: BaseModel](Protocol):
    def handle_response(
        self,
        item: ItemT,
        state: SequenceState,
        response: CompletionResult[T],
    ) -> ItemProcessingResult[ItemT, T]: ...


class StructuredOutputResponseHandler[ItemT: BaseDataItem, T: BaseModel](
    ResponseHandler[ItemT, T]
):
    """Extract structured output and carry an optional ``context`` attribute forward."""

    @override
    def handle_response(
        self,
        item: ItemT,
        state: SequenceState,
        response: CompletionResult[T],
    ) -> ItemProcessingResult[ItemT, T]:
        output = response.output.structured_response
        if output is None:
            raise ValueError(
                "LLM returned no structured response, received response: "
                f"{response.output.text_response}"
            )

        if isinstance(item, PredictionDataItem):
            item.predictions = output

        updated_state = SequenceState(
            conversation=response.updated_conversation,
            domain_context=getattr(output, "context", None),
            items_processed=state.items_processed,
            current_item_index=state.current_item_index,
        )
        return ItemProcessingResult(state=updated_state, item=item, response=output)


class PredictionsRefinementResponseHandler[ItemT: PredictionDataItem, T: BaseModel](
    StructuredOutputResponseHandler[ItemT, T]
):
    """Compatibility name for prediction-refinement workflows."""


@final
class TextOnlyRequestHandler(RequestHandler[Never]):
    @override
    def handle_request(self, state: SequenceState) -> CompletionRequest[Never]:
        return CompletionRequest(input=state.conversation, structured_output=None)


@final
class TextExtractionResponseHandler[ItemT: BaseDataItem](ResponseHandler[ItemT, Never]):
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
        self.llm_engine = llm_engine
        self.request_handler = request_handler
        self.response_handler = response_handler

    def process(
        self,
        item: ItemT,
        state: SequenceState,
    ) -> ItemProcessingResult[ItemT, OutputT]:
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
        request = self.request_handler.handle_request(state)
        logger.debug(
            "Processing item async",
            item_id=item.metadata.sample_id if item.metadata else None,
            filename=item.metadata.filename if item.metadata else None,
        )
        result = await self.llm_engine.process_async(request)
        return self.response_handler.handle_response(item, state, result)

