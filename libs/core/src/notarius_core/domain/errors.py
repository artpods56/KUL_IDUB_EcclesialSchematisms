class NotariusCoreError(Exception):
    """Base application error."""


class NotFoundError(NotariusCoreError):
    """A requested resource was not found."""

    def __init__(self, resource: str, resource_id: str) -> None:
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} not found: {resource_id}")


class ConflictError(NotariusCoreError):
    """A conflicting resource already exists."""


class ForbiddenError(NotariusCoreError):
    """Access to a resource is denied."""


class ValidationError(NotariusCoreError):
    """Request data failed validation."""


# --- storage exceptions ---

class StorageException(NotariusCoreError):
    """Base for storage related exceptions"""

class ObjectAlreadyExistsError(StorageException):
    """Raised when an object already exists in the storage backend."""