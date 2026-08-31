from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from grafy_core.application.templates import TemplateService
from grafy_core.domain.errors import (
    CollaborationCommandRejectedError,
    Failure,
    FailureKind,
    FailureSpec,
    NotFoundError,
)
from grafy_core.domain.identity import ActorContext
from grafy_core.domain.templates import (
    TemplateCopyRejectedError,
    TemplateLibraryError,
    TemplateUnavailableError,
)


class _TemplateRepository:
    def __init__(self, template: object | None = None) -> None:
        self.template = template

    async def get(self, _workspace_id: object, _template_id: object) -> object | None:
        return self.template


class _GraphRepository:
    async def get_revision(
        self,
        _workspace_id: object,
        _graph_id: object,
        _revision: int,
    ) -> object:
        return SimpleNamespace(document=object(), name="Source graph")


class _TemplateUnitOfWork:
    def __init__(self, template: object | None = None) -> None:
        self.identity = object()
        self.graphs = _GraphRepository()
        self.templates = _TemplateRepository(template)

    async def __aenter__(self) -> "_TemplateUnitOfWork":
        return self

    async def __aexit__(
        self,
        _exc_type: object,
        _exc: object,
        _traceback: object,
    ) -> None:
        return None


def test_declared_failure_keeps_diagnostics_out_of_public_message() -> None:
    resource_id = str(uuid4())

    error = NotFoundError("Workspace", resource_id)

    assert resource_id in str(error)
    assert error.public_message == "Not found"
    assert error.failure_spec is not None
    assert error.failure_spec.code == "resource.not_found"
    assert error.failure_spec.kind is FailureKind.NOT_FOUND


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("not_namespaced", "Safe message"),
        ("Graph.NotFound", "Safe message"),
        ("graph.not-found", "Safe message"),
        ("graph.not_found", "   "),
        ("graph.not_found", "x" * 201),
    ],
)
def test_failure_spec_rejects_unstable_or_unsafe_values(
    code: str,
    message: str,
) -> None:
    with pytest.raises(ValueError):
        FailureSpec(
            code=code,
            kind=FailureKind.VALIDATION,
            public_message=message,
        )


def test_failure_is_frozen_and_rejects_extra_fields() -> None:
    failure = Failure(
        error_id=uuid4(),
        code="graph.not_found",
        kind=FailureKind.NOT_FOUND,
        message="Not found",
    )

    with pytest.raises(ValidationError):
        Failure.model_validate({**failure.model_dump(), "diagnostic": "secret"})
    with pytest.raises(ValidationError):
        failure.message = "Changed"  # type: ignore[misc]


def test_template_library_validation_is_not_declared_public() -> None:
    error = TemplateLibraryError("Template name contains internal detail")

    assert not hasattr(error, "public_message")


async def test_template_copy_rejection_preserves_typed_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application_module = __import__(
        "grafy_core.application.templates",
        fromlist=["authorize_workspace"],
    )
    monkeypatch.setattr(application_module, "authorize_workspace", AsyncMock())

    def reject_copy(_document: object) -> object:
        raise CollaborationCommandRejectedError(
            code="secret_binding",
            message="Node abc contains a secret binding",
        )

    monkeypatch.setattr(
        application_module,
        "sanitize_document_for_cross_workspace_copy",
        reject_copy,
    )
    service = TemplateService(lambda: _TemplateUnitOfWork())  # type: ignore[arg-type]

    with pytest.raises(TemplateCopyRejectedError) as raised:
        await service.create_from_graph_revision(
            actor=ActorContext(user_id=uuid4()),
            workspace_id=uuid4(),
            source_graph_id=uuid4(),
            source_revision=1,
            name="Template",
            description=None,
        )

    assert raised.value.reason_code == "secret_binding"
    assert "Node abc" in str(raised.value)
    assert raised.value.public_message == "This graph cannot be saved as a template"
    assert isinstance(raised.value.__cause__, CollaborationCommandRejectedError)


async def test_archived_template_has_typed_unavailable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    application_module = __import__(
        "grafy_core.application.templates",
        fromlist=["authorize_workspaces"],
    )
    monkeypatch.setattr(application_module, "authorize_workspaces", AsyncMock())
    archived_template = SimpleNamespace(is_available=False)
    service = TemplateService(  # type: ignore[arg-type]
        lambda: _TemplateUnitOfWork(archived_template)
    )

    with pytest.raises(TemplateUnavailableError) as raised:
        await service.instantiate(
            actor=ActorContext(user_id=uuid4()),
            source_workspace_id=uuid4(),
            template_id=uuid4(),
            destination_workspace_id=uuid4(),
            name="Graph",
            folder_id=None,
        )

    assert raised.value.public_message == "Archived templates cannot be used"
    assert raised.value.failure_spec is not None
    assert raised.value.failure_spec.code == "template.unavailable"
