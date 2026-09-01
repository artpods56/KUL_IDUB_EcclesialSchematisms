import asyncio
from uuid import UUID

from .errors import ExecutionTimeoutError
from .models import ExecutionState
from .transport import HttpTransport


class ExecutionHandle:
    def __init__(
        self,
        *,
        transport: HttpTransport,
        workspace_id: UUID,
        state: ExecutionState,
    ) -> None:
        self._transport = transport
        self._workspace_id = workspace_id
        self._state = state

    @property
    def execution_id(self) -> UUID:
        return self._state.execution_id

    @property
    def status(self) -> str:
        return self._state.status

    async def get(self) -> ExecutionState:
        payload = await self._transport.request_json(
            operation=f"get graph execution {self.execution_id}",
            method="GET",
            path=(
                f"/v1/workspaces/{self._workspace_id}/executions/"
                f"{self.execution_id}"
            ),
        )
        self._state = ExecutionState.model_validate(payload)
        return self._state

    async def wait(
        self,
        *,
        timeout: float,
        poll_interval: float = 0.1,
    ) -> ExecutionState:
        if timeout <= 0:
            raise ValueError("Execution wait timeout must be positive")
        if poll_interval < 0:
            raise ValueError("Execution poll interval must not be negative")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not self._state.terminal:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise ExecutionTimeoutError(
                    operation=f"wait for graph execution {self.execution_id}",
                    detail=(
                        f"Timed out after {timeout:g} seconds; last status was "
                        f"{self._state.status!r}"
                    ),
                )
            if poll_interval > 0:
                await asyncio.sleep(min(poll_interval, remaining))
            await self.get()
        return self._state


__all__ = ["ExecutionHandle"]
