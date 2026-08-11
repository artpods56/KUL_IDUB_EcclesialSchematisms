"""Copy-based graph templates owned by a workspace."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from notarius_core.domain.saved_graphs import SavedGraphDocument


class TemplateState(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class TemplateLibraryError(ValueError):
    """Raised when a template library invariant would be violated."""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _required_text(value: str, label: str, maximum: int) -> str:
    normalized = value.strip()
    if normalized == "":
        raise TemplateLibraryError(f"{label} must not be blank")
    if len(normalized) > maximum:
        raise TemplateLibraryError(f"{label} must be at most {maximum} characters")
    return normalized


def _optional_description(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized == "":
        return None
    if len(normalized) > 1_000:
        raise TemplateLibraryError(
            "Template description must be at most 1000 characters"
        )
    return normalized


@dataclass
class Template:
    """Immutable graph snapshot plus independently editable library metadata."""

    workspace_id: UUID
    source_graph_id: UUID
    source_revision: int
    source_graph_name: str
    snapshot_document: SavedGraphDocument
    name: str
    created_by_user_id: UUID | None = None
    description: str | None = None
    state: TemplateState = TemplateState.ACTIVE
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if isinstance(self.source_revision, bool) or self.source_revision < 1:
            raise TemplateLibraryError(
                "Template source revision must be a positive integer"
            )
        self.source_graph_name = _required_text(
            self.source_graph_name,
            "Template source graph name",
            160,
        )
        self.name = _required_text(self.name, "Template name", 160)
        self.description = _optional_description(self.description)
        self.state = TemplateState(self.state)
        if self.created_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise TemplateLibraryError("Template timestamps must be timezone-aware")

    @property
    def is_available(self) -> bool:
        return self.state is TemplateState.ACTIVE

    @property
    def node_count(self) -> int:
        return len(self.snapshot_document.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.snapshot_document.edges)

    def update_metadata(
        self,
        *,
        name: str,
        description: str | None,
        when: datetime | None = None,
    ) -> None:
        self.name = _required_text(name, "Template name", 160)
        self.description = _optional_description(description)
        self.updated_at = when or _utc_now()

    def archive(self, *, when: datetime | None = None) -> None:
        self.state = TemplateState.ARCHIVED
        self.updated_at = when or _utc_now()


__all__ = ["Template", "TemplateLibraryError", "TemplateState"]
