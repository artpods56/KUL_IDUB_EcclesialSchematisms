"""Process-wide admission budget for top-level graph executions."""

from dataclasses import dataclass, field
from threading import Lock
from typing import Literal
from uuid import UUID, uuid4


class RunExecutionCapacityError(RuntimeError):
    """Raised before admission when the process execution budget is exhausted."""

    error_code: Literal["execution_capacity_exceeded"] = (
        "execution_capacity_exceeded"
    )

    def __init__(self, max_active_executions: int) -> None:
        self.max_active_executions = max_active_executions
        super().__init__(
            "Run execution capacity is exhausted; "
            f"the process already owns {max_active_executions} active executions"
        )


@dataclass(frozen=True, slots=True)
class ExecutionAdmissionLease:
    _limiter: "ExecutionAdmissionLimiter" = field(repr=False)
    _lease_id: UUID = field(repr=False)

    def release(self) -> None:
        self._limiter.release(self._lease_id)


class ExecutionAdmissionLimiter:
    """Reject top-level work above a shared process-wide execution budget."""

    def __init__(self, max_active_executions: int) -> None:
        if max_active_executions < 1:
            raise ValueError("Maximum active executions must be at least one")
        self._max_active_executions = max_active_executions
        self._active_lease_ids: set[UUID] = set()
        self._lock = Lock()

    def acquire(self) -> ExecutionAdmissionLease:
        with self._lock:
            if len(self._active_lease_ids) >= self._max_active_executions:
                raise RunExecutionCapacityError(self._max_active_executions)
            lease_id = uuid4()
            self._active_lease_ids.add(lease_id)
        return ExecutionAdmissionLease(self, lease_id)

    def release(self, lease_id: UUID) -> None:
        with self._lock:
            self._active_lease_ids.discard(lease_id)


__all__ = [
    "ExecutionAdmissionLease",
    "ExecutionAdmissionLimiter",
    "RunExecutionCapacityError",
]
