from notarius_core.application.sequence_state import SequenceState


def test_sequence_state_accepts_generic_domain_context() -> None:
    state = SequenceState.empty()
    updated = SequenceState(
        conversation=state.conversation,
        domain_context={"active": "context"},
        items_processed=1,
        current_item_index=2,
    )

    assert updated.domain_context == {"active": "context"}
    assert updated.current_item_index == 2

