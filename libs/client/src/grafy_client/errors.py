class GrafyClientError(RuntimeError):
    def __init__(
        self,
        *,
        operation: str,
        detail: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.operation = operation
        self.detail = detail
        self.status_code = status_code
        self.request_id = request_id

        context = operation
        if status_code is not None:
            context += f" failed with HTTP {status_code}"
        else:
            context += " failed"
        if request_id is not None:
            context += f" (request {request_id})"
        super().__init__(f"{context}: {detail}")


class ExecutionTimeoutError(GrafyClientError):
    pass


__all__ = ["ExecutionTimeoutError", "GrafyClientError"]
