from pydantic import BaseModel

from notarius_core.application.processors.item_processor import (
    ItemProcessor,
    StandardRequestHandler,
    StructuredOutputResponseHandler,
)
from notarius_core.application.sequence_state import SequenceState
from notarius_core.domain.models.completions import BaseProviderResponse
from notarius_core.domain.models.dataset import PredictionDataItem
from notarius_core.ports.llm import CompletionRequest, CompletionResult


class Output(BaseModel):
    value: str
    context: dict[str, str]


class ProviderResponse(BaseProviderResponse[Output]):
    structured_response: Output | None
    text_response: str | None


class FakeEngine:
    stats = {}

    def process(self, request: CompletionRequest[Output]) -> CompletionResult[Output]:
        return CompletionResult(
            output=ProviderResponse(
                structured_response=Output(value="ok", context={"next": "ctx"}),
                text_response=None,
            ),
            conversation=request.input,
        )

    async def process_async(
        self,
        request: CompletionRequest[Output],
    ) -> CompletionResult[Output]:
        return self.process(request)


def test_item_processor_sets_prediction_and_context() -> None:
    item = PredictionDataItem(image_path=None, text="input")
    processor = ItemProcessor(
        llm_engine=FakeEngine(),
        request_handler=StandardRequestHandler(Output),
        response_handler=StructuredOutputResponseHandler(),
    )

    result = processor.process(item, SequenceState.empty())

    assert result.item.predictions == Output(value="ok", context={"next": "ctx"})
    assert result.state.domain_context == {"next": "ctx"}

