from uuid import UUID, uuid4

import pytest

from grafy_core.domain.module_library import (
    Module,
    ModuleLibraryError,
    ModulePublicationState,
)


def test_publish_deprecate_withdraw_state_machine() -> None:
    module = Module(
        workspace_id=UUID(int=1),
        source_graph_id=UUID(int=2),
        name="Capitalize",
    )
    module.apply_publish(revision=1)
    assert module.publication_state == ModulePublicationState.PUBLISHED
    assert module.current_library_release == 1
    assert module.is_listed_in_library

    module.deprecate()
    assert module.publication_state == ModulePublicationState.DEPRECATED
    assert module.is_listed_in_library

    module.apply_publish(revision=2, name="Capitalize v2")
    assert module.publication_state == ModulePublicationState.PUBLISHED
    assert module.current_library_release == 2
    assert module.name == "Capitalize v2"

    module.withdraw()
    assert module.publication_state == ModulePublicationState.WITHDRAWN
    assert not module.is_listed_in_library

    module.apply_publish(revision=3)
    assert module.publication_state == ModulePublicationState.PUBLISHED


def test_withdrawn_module_cannot_be_deprecated() -> None:
    module = Module(
        workspace_id=uuid4(),
        source_graph_id=uuid4(),
        name="Legacy",
    )
    module.apply_publish(revision=1)
    module.withdraw()
    with pytest.raises(ModuleLibraryError, match="cannot be deprecated"):
        module.deprecate()
