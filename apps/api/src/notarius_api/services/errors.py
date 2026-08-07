class WorkbenchOperationError(RuntimeError):
    """Expected application error that a workbench route can render."""


class ArtifactContentUnavailableError(WorkbenchOperationError):
    """An artifact exists, but its persisted content cannot be read safely."""


__all__ = ["ArtifactContentUnavailableError", "WorkbenchOperationError"]
