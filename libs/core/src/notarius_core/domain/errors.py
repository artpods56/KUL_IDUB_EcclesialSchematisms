from uuid import UUID


class NotariusCoreError(Exception):
    """Base application error."""


class NotFoundError(NotariusCoreError):
    """A requested resource was not found."""

    def __init__(self, resource: str, resource_id: str) -> None:
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} not found: {resource_id}")


class ObjectAlreadyExistsError(NotariusCoreError):
    """Raised when an object already exists in the storage backend."""


class ConcurrentWriteError(NotariusCoreError):
    """Raised when persistence detects an optimistic concurrency conflict."""


class IdentityInvariantError(NotariusCoreError):
    """Raised when an identity or workspace invariant would be violated."""


class BootstrapOwnerRequiredError(IdentityInvariantError):
    """Raised while the sealed legacy workspace is awaiting its owner."""


class BootstrapOwnerMismatchError(IdentityInvariantError):
    """Raised when a login does not match the configured bootstrap identity."""


class LastWorkspaceOwnerError(IdentityInvariantError):
    """Raised when an operation would remove the last active workspace owner."""


class UserDisabledError(IdentityInvariantError):
    """Raised when a disabled user attempts to authenticate or authorize."""


class CapabilityDeniedError(IdentityInvariantError):
    """Raised when an active membership lacks one required capability."""

    def __init__(self, *, capability: str, workspace_id: UUID, user_id: UUID) -> None:
        self.capability = capability
        self.workspace_id = workspace_id
        self.user_id = user_id
        super().__init__(
            f"User {user_id} is not authorized for capability {capability!r} "
            f"in workspace {workspace_id}"
        )


class SavedGraphRevisionConflictError(NotariusCoreError):
    def __init__(
        self,
        *,
        graph_id: UUID,
        expected_revision: int,
        actual_revision: int | None,
    ) -> None:
        self.graph_id = graph_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        if actual_revision is None:
            detail = "the graph changed while it was being saved"
        else:
            detail = f"the current revision is {actual_revision}"
        super().__init__(
            f"Saved graph {graph_id} revision conflict: expected "
            f"{expected_revision}, but {detail}"
        )
