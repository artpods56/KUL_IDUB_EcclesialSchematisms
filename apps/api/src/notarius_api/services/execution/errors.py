from notarius_api.services.errors import WorkbenchOperationError


class GraphExecutionError(WorkbenchOperationError):
    pass


__all__ = ["GraphExecutionError"]
