from collections.abc import Mapping, Sequence
from types import TracebackType
from unittest.mock import patch

import pytest
from pydantic import SecretStr
from sqlalchemy import TextClause, URL

from notarius_plugin_sql.models import SqlStatement, SqlValue
from notarius_plugin_sql.nodes import (
    ExecuteSqlConfig,
    PostgresSslMode,
    SqlExecutionError,
)
from notarius_plugin_sql.sqlalchemy import SqlAlchemyPostgresBatchExecutor


class FakeResult:
    def __init__(
        self,
        *,
        columns: tuple[str, ...] = (),
        rows: Sequence[Sequence[object]] = (),
        rowcount: int = -1,
    ) -> None:
        self._columns = columns
        self._rows = rows
        self.rowcount = rowcount
        self.fetch_sizes: list[int] = []

    @property
    def returns_rows(self) -> bool:
        return bool(self._columns)

    def keys(self) -> Sequence[str]:
        return self._columns

    def fetchmany(self, size: int) -> Sequence[Sequence[object]]:
        self.fetch_sizes.append(size)
        return self._rows[:size]


class FakeConnection:
    def __init__(self, results: list[FakeResult | Exception]) -> None:
        self._results = results
        self.executions: list[tuple[str, dict[str, SqlValue]]] = []

    async def execute(
        self,
        statement: TextClause,
        parameters: Mapping[str, SqlValue],
    ) -> FakeResult:
        self.executions.append((statement.text, dict(parameters)))
        result = self._results[len(self.executions) - 1]
        if isinstance(result, Exception):
            raise result
        return result


class FakeBegin:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> FakeConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc, traceback
        if exc_type is None:
            self.committed = True
        else:
            self.rolled_back = True


class FakeEngine:
    def __init__(self, connection: FakeConnection) -> None:
        self.transaction = FakeBegin(connection)
        self.disposed = False

    def begin(self) -> FakeBegin:
        return self.transaction

    async def dispose(self) -> None:
        self.disposed = True


class FakeEngineFactory:
    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine
        self.url: str | None = None
        self.options: dict[str, object] | None = None

    def __call__(self, url: URL, **options: object) -> FakeEngine:
        self.url = url.render_as_string(hide_password=False)
        self.options = options
        return self._engine


def postgres_config() -> ExecuteSqlConfig:
    return ExecuteSqlConfig(
        host="postgres.internal",
        port=5433,
        database="documents",
        username="notarius",
        ssl_mode=PostgresSslMode.VERIFY_FULL,
        timeout_seconds=12.5,
    )


async def test_sqlalchemy_executor_runs_named_batch_in_order_and_commits() -> None:
    connection = FakeConnection(
        [
            FakeResult(rowcount=1),
            FakeResult(columns=("id", "name"), rows=((4, "invoice"),)),
        ]
    )
    engine = FakeEngine(connection)
    factory = FakeEngineFactory(engine)

    with patch("notarius_plugin_sql.sqlalchemy.create_async_engine", factory):
        results = await SqlAlchemyPostgresBatchExecutor().execute(
            postgres_config(),
            SecretStr("postgres-password"),
            [
                SqlStatement(
                    sql="insert into jobs(name) values(:name)",
                    parameters={"name": "invoice"},
                ),
                SqlStatement(sql="select id, name from jobs"),
            ],
        )

    assert factory.url == (
        "postgresql+asyncpg://notarius:postgres-password@"
        "postgres.internal:5433/documents"
    )
    assert factory.options == {
        "connect_args": {
            "ssl": "verify-full",
            "timeout": 12.5,
            "command_timeout": 12.5,
        },
        "hide_parameters": True,
        "pool_pre_ping": True,
        "pool_timeout": 12.5,
    }
    assert connection.executions == [
        ("insert into jobs(name) values(:name)", {"name": "invoice"}),
        ("select id, name from jobs", {}),
    ]
    assert [result.model_dump(mode="json") for result in results] == [
        {
            "statement_index": 0,
            "returns_rows": False,
            "table": {"columns": [], "rows": []},
            "affected_rows": 1,
        },
        {
            "statement_index": 1,
            "returns_rows": True,
            "table": {
                "columns": [
                    {"id": "column_1", "title": "id", "value_type": "integer"},
                    {"id": "column_2", "title": "name", "value_type": "text"},
                ],
                "rows": [{"column_1": 4, "column_2": "invoice"}],
            },
            "affected_rows": None,
        },
    ]
    assert engine.transaction.committed
    assert not engine.transaction.rolled_back
    assert engine.disposed


async def test_sqlalchemy_executor_rolls_back_and_reports_statement_index() -> None:
    failure = RuntimeError("unique constraint violation")
    connection = FakeConnection(
        [
            FakeResult(columns=("value",), rows=((1,),)),
            failure,
        ]
    )
    engine = FakeEngine(connection)

    with patch(
        "notarius_plugin_sql.sqlalchemy.create_async_engine",
        FakeEngineFactory(engine),
    ):
        with pytest.raises(
            SqlExecutionError,
            match="statement at index 1 failed",
        ) as captured:
            await SqlAlchemyPostgresBatchExecutor().execute(
                postgres_config(),
                SecretStr("password"),
                [
                    SqlStatement(sql="select 1"),
                    SqlStatement(sql="insert duplicate"),
                ],
            )

    assert captured.value.__cause__ is failure
    assert not engine.transaction.committed
    assert engine.transaction.rolled_back
    assert engine.disposed


async def test_sqlalchemy_executor_rolls_back_when_result_exceeds_row_limit() -> None:
    oversized = FakeResult(
        columns=("value",),
        rows=((1,), (2,), (3,)),
    )
    connection = FakeConnection([oversized])
    engine = FakeEngine(connection)
    config = postgres_config().model_copy(update={"max_result_rows": 2})

    with patch(
        "notarius_plugin_sql.sqlalchemy.create_async_engine",
        FakeEngineFactory(engine),
    ):
        with pytest.raises(
            SqlExecutionError,
            match="statement at index 0 exceeded the configured 2-row result limit",
        ):
            await SqlAlchemyPostgresBatchExecutor().execute(
                config,
                SecretStr("password"),
                [SqlStatement(sql="select value from too_many_rows")],
            )

    assert oversized.fetch_sizes == [3]
    assert not engine.transaction.committed
    assert engine.transaction.rolled_back
    assert engine.disposed
