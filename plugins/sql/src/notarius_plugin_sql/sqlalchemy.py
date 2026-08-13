import base64
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import date, datetime, time
from decimal import Decimal
from typing import Protocol, cast, final, override
from uuid import UUID

from pydantic import SecretStr
from sqlalchemy import URL, TextClause, text
from sqlalchemy.ext.asyncio import create_async_engine

from notarius_core.operators.tables import (
    Table,
    TableColumn,
    TableValue,
    TableValueType,
)

from notarius_plugin_sql.models import SqlResult, SqlStatement, SqlValue
from notarius_plugin_sql.nodes import (
    ExecuteSqlConfig,
    PostgresBatchExecutor,
    SqlExecutionError,
)


class _SqlAlchemyResult(Protocol):
    @property
    def returns_rows(self) -> bool: ...

    @property
    def rowcount(self) -> int: ...

    def keys(self) -> Sequence[str]: ...

    def fetchmany(self, size: int) -> Sequence[Sequence[object]]: ...


class _SqlAlchemyConnection(Protocol):
    async def execute(
        self,
        statement: TextClause,
        parameters: Mapping[str, SqlValue],
    ) -> _SqlAlchemyResult: ...


class _SqlAlchemyEngine(Protocol):
    def begin(self) -> AbstractAsyncContextManager[_SqlAlchemyConnection]: ...

    async def dispose(self) -> None: ...


@final
class SqlAlchemyPostgresBatchExecutor(PostgresBatchExecutor):
    @override
    async def execute(
        self,
        config: ExecuteSqlConfig,
        password: SecretStr,
        statements: list[SqlStatement],
        /,
    ) -> list[SqlResult]:
        url = URL.create(
            drivername="postgresql+asyncpg",
            username=config.username,
            password=password.get_secret_value(),
            host=config.host,
            port=config.port,
            database=config.database,
        )
        engine = cast(
            _SqlAlchemyEngine,
            create_async_engine(
                url,
                connect_args={
                    "ssl": config.ssl_mode.value,
                    "timeout": config.timeout_seconds,
                    "command_timeout": config.timeout_seconds,
                },
                hide_parameters=True,
                pool_pre_ping=True,
                pool_timeout=config.timeout_seconds,
            ),
        )
        try:
            results: list[SqlResult] = []
            async with engine.begin() as connection:
                for index, statement in enumerate(statements):
                    try:
                        result = await connection.execute(
                            text(statement.sql),
                            statement.parameters,
                        )
                        returns_rows = result.returns_rows
                        rows = (
                            result.fetchmany(config.max_result_rows + 1)
                            if returns_rows
                            else ()
                        )
                        if len(rows) > config.max_result_rows:
                            raise SqlExecutionError(
                                "PostgreSQL statement at index "
                                f"{index} exceeded the configured "
                                f"{config.max_result_rows}-row result limit"
                            )
                        affected_rows = None
                        if not returns_rows and result.rowcount >= 0:
                            affected_rows = result.rowcount
                        column_titles = [str(name) for name in result.keys()]
                        column_ids = [
                            f"column_{column_index + 1}"
                            for column_index in range(len(column_titles))
                        ]
                        table_rows = [
                            {
                                column_id: _table_value(value)
                                for column_id, value in zip(
                                    column_ids,
                                    row,
                                    strict=True,
                                )
                            }
                            for row in rows
                        ]
                        results.append(
                            SqlResult(
                                statement_index=index,
                                returns_rows=returns_rows,
                                table=Table(
                                    columns=[
                                        TableColumn(
                                            id=column_id,
                                            title=title,
                                            value_type=_column_value_type(
                                                [row[column_index] for row in rows]
                                            ),
                                        )
                                        for column_index, (
                                            column_id,
                                            title,
                                        ) in enumerate(
                                            zip(
                                                column_ids,
                                                column_titles,
                                                strict=True,
                                            )
                                        )
                                    ],
                                    rows=table_rows,
                                ),
                                affected_rows=affected_rows,
                            )
                        )
                    except SqlExecutionError:
                        raise
                    except Exception as exc:
                        raise SqlExecutionError(
                            f"PostgreSQL statement at index {index} failed"
                        ) from exc
            return results
        finally:
            await engine.dispose()


def _table_value(value: object) -> TableValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date | time | UUID):
        return value.isoformat() if not isinstance(value, UUID) else str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, list | tuple):
        return [_table_value(item) for item in cast(Sequence[object], value)]
    if isinstance(value, dict):
        return {
            str(key): _table_value(item)
            for key, item in cast(dict[object, object], value).items()
        }
    raise SqlExecutionError(
        f"PostgreSQL returned unsupported value type {type(value).__name__}"
    )


def _column_value_type(values: Sequence[object]) -> TableValueType:
    inferred = {_value_type(value) for value in values if value is not None}
    if not inferred:
        return TableValueType.UNKNOWN
    if inferred <= {TableValueType.INTEGER, TableValueType.NUMBER}:
        return (
            TableValueType.INTEGER
            if inferred == {TableValueType.INTEGER}
            else TableValueType.NUMBER
        )
    if len(inferred) == 1:
        return next(iter(inferred))
    return TableValueType.MIXED


def _value_type(value: object) -> TableValueType:
    if isinstance(value, bool):
        return TableValueType.BOOLEAN
    if isinstance(value, int):
        return TableValueType.INTEGER
    if isinstance(value, float):
        return TableValueType.NUMBER
    if isinstance(value, Decimal):
        return TableValueType.DECIMAL
    if isinstance(value, datetime):
        return TableValueType.DATETIME
    if isinstance(value, date):
        return TableValueType.DATE
    if isinstance(value, bytes):
        return TableValueType.BINARY
    if isinstance(value, list | dict):
        return TableValueType.JSON
    if isinstance(value, str):
        return TableValueType.TEXT
    return TableValueType.UNKNOWN
