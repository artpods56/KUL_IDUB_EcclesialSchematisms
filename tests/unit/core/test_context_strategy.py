from notarius_core.application.context.provider import PageContentContextProvider
from notarius_core.application.sequence_state import SequenceState
from notarius_core.domain.models.dataset import BaseDataItem


def test_page_content_context_provider_returns_none_for_missing_lookahead() -> None:
    items = [BaseDataItem(image_path=None, text="first")]
    provider = PageContentContextProvider[BaseDataItem](offset=1)

    context = provider.get_context(items, SequenceState.empty())

    assert context == {"NEXT_PAGE__TEXT": None}

