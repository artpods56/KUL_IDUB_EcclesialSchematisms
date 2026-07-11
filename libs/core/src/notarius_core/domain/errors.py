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
