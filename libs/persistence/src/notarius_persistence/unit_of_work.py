from types import TracebackType
from typing import override

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm.exc import StaleDataError

from notarius_core.domain.errors import ConcurrentWriteError
from notarius_core.ports.saved_graphs import (
    SavedGraphRepositoryPort,
    SavedGraphUnitOfWorkPort,
)

from notarius_persistence.adapters.repositories import SqlSavedGraphRepository


class SqlAlchemySavedGraphUnitOfWork(SavedGraphUnitOfWorkPort):
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._graphs: SavedGraphRepositoryPort | None = None

    @property
    @override
    def graphs(self) -> SavedGraphRepositoryPort:
        if self._graphs is None:
            raise RuntimeError("Unit of work is not entered")
        return self._graphs

    @override
    async def __aenter__(self) -> "SqlAlchemySavedGraphUnitOfWork":
        if self._session is not None:
            raise RuntimeError("Unit of work is already entered")
        self._session = self._session_factory()
        self._graphs = SqlSavedGraphRepository(self._session)
        return self

    @override
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        if self._session is None:
            return
        if exc_type is not None:
            await self.rollback()
        await self._session.close()
        self._session = None
        self._graphs = None

    @override
    async def commit(self) -> None:
        if self._session is None:
            raise RuntimeError("Unit of work is not entered")
        try:
            await self._session.commit()
        except StaleDataError as exc:
            await self._session.rollback()
            raise ConcurrentWriteError(
                "The saved graph changed in another transaction"
            ) from exc

    @override
    async def rollback(self) -> None:
        if self._session is None:
            raise RuntimeError("Unit of work is not entered")
        await self._session.rollback()
