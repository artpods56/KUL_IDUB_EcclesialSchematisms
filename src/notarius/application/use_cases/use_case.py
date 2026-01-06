"""
Base classes for use cases in the application layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BaseRequest(ABC): ...


@dataclass
class BaseResponse(ABC): ...


class BaseUseCase[TRequest: BaseRequest, TResponse: BaseResponse](ABC):
    """
    Base class for all use cases following the Command Handler pattern.

    Use cases orchestrate domain services and infrastructure components
    to implement business workflows.
    """

    @abstractmethod
    def execute(self, request: TRequest) -> TResponse:
        """Execute the use case with the given request."""
        pass


class AsyncBaseUseCase[TRequest: BaseRequest, TResponse: BaseResponse](ABC):
    """
    Async base class for use cases that require asynchronous execution.

    Use this for use cases that involve I/O-bound operations like
    API calls, database queries, or file operations that benefit
    from concurrent execution.
    """

    @abstractmethod
    async def execute(self, request: TRequest) -> TResponse:
        """Execute the use case asynchronously with the given request."""
        pass
