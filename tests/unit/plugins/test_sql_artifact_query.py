import asyncio

import pytest

from grafy_core.table_contracts import (
    Table,
    TableColumn,
    TableValue,
    TableValueType,
)
from grafy_plugin_sql.artifact_query import (
    ArtifactQueryExecutorError,
    IsolatedDuckDbArtifactTableExecutor,
)
from grafy_plugin_sql.models import SqlStatement


def relation(
    columns: list[tuple[str, TableValueType]],
    rows: list[dict[str, TableValue]],
) -> Table:
    return Table(
        columns=[
            TableColumn(id=column_id, title=column_id, value_type=value_type)
            for column_id, value_type in columns
        ],
        rows=rows,
    )


async def test_artifact_query_joins_relations_and_returns_one_table_per_statement() -> (
    None
):
    parcels = relation(
        [
            ("id", TableValueType.INTEGER),
            ("owner_id", TableValueType.INTEGER),
            ("area", TableValueType.NUMBER),
        ],
        [
            {"id": 1, "owner_id": 10, "area": 40.5},
            {"id": 2, "owner_id": 20, "area": 75.0},
            {"id": 3, "owner_id": 10, "area": 100.0},
        ],
    )
    owners = relation(
        [("id", TableValueType.INTEGER), ("name", TableValueType.TEXT)],
        [{"id": 10, "name": "Ada"}, {"id": 20, "name": "Grace"}],
    )

    tables = await IsolatedDuckDbArtifactTableExecutor().execute(
        [
            SqlStatement(
                sql=(
                    "select p.id, o.name, p.area "
                    "from parcels p join owners o on o.id = p.owner_id "
                    "where p.area >= :minimum_area order by p.id"
                ),
                parameters={"minimum_area": 70},
            ),
            SqlStatement(
                sql=(
                    "select owner_id, sum(area) as total_area "
                    "from parcels group by owner_id order by owner_id"
                )
            ),
        ],
        ["parcels", "owners"],
        [parcels, owners],
    )

    assert [column.title for column in tables[0].columns] == ["id", "name", "area"]
    assert tables[0].rows == [
        {"column_1": 2, "column_2": "Grace", "column_3": 75.0},
        {"column_1": 3, "column_2": "Ada", "column_3": 100.0},
    ]
    assert [column.title for column in tables[1].columns] == [
        "owner_id",
        "total_area",
    ]
    assert tables[1].rows == [
        {"column_1": 10, "column_2": 140.5},
        {"column_1": 20, "column_2": 75.0},
    ]


async def test_artifact_query_translates_only_canonical_parameter_placeholders() -> (
    None
):
    source = relation(
        [("id", TableValueType.INTEGER)],
        [{"id": 1}],
    )

    tables = await IsolatedDuckDbArtifactTableExecutor().execute(
        [
            SqlStatement(
                sql=(
                    "select ':literal' as literal, id::BIGINT as id, "
                    ":bound as bound from source -- :comment"
                ),
                parameters={"bound": 42},
            )
        ],
        ["source"],
        [source],
    )

    assert tables[0].rows == [{"column_1": ":literal", "column_2": 1, "column_3": 42}]


async def test_artifact_query_preserves_canonical_table_value_types() -> None:
    source = Table(
        columns=[
            TableColumn(
                id="amount",
                title="Amount",
                value_type=TableValueType.DECIMAL,
            ),
            TableColumn(
                id="day",
                title="Day",
                value_type=TableValueType.DATE,
            ),
            TableColumn(
                id="observed_at",
                title="Observed at",
                value_type=TableValueType.DATETIME,
            ),
            TableColumn(
                id="attributes",
                title="Attributes",
                value_type=TableValueType.JSON,
            ),
            TableColumn(
                id="payload",
                title="Payload",
                value_type=TableValueType.BINARY,
            ),
            TableColumn(
                id="mixed",
                title="Mixed",
                value_type=TableValueType.MIXED,
            ),
        ],
        rows=[
            {
                "amount": "12.50",
                "day": "2026-07-23",
                "observed_at": "2026-07-23T09:30:00+00:00",
                "attributes": {"tags": ["surveyed"]},
                "payload": "AP8=",
                "mixed": 7,
            },
            {
                "amount": None,
                "day": None,
                "observed_at": None,
                "attributes": [1, 2],
                "payload": None,
                "mixed": "seven",
            },
        ],
    )

    tables = await IsolatedDuckDbArtifactTableExecutor().execute(
        [SqlStatement(sql="select * from source")],
        ["source"],
        [source],
    )

    assert [column.value_type for column in tables[0].columns] == [
        TableValueType.DECIMAL,
        TableValueType.DATE,
        TableValueType.DATETIME,
        TableValueType.JSON,
        TableValueType.BINARY,
        TableValueType.MIXED,
    ]
    assert tables[0].rows == [
        {
            "column_1": "12.50",
            "column_2": "2026-07-23",
            "column_3": "2026-07-23T09:30:00+00:00",
            "column_4": {"tags": ["surveyed"]},
            "column_5": "AP8=",
            "column_6": 7,
        },
        {
            "column_1": None,
            "column_2": None,
            "column_3": None,
            "column_4": [1, 2],
            "column_5": None,
            "column_6": "seven",
        },
    ]


@pytest.mark.parametrize(
    ("sql", "error_kind"),
    [
        ("select 1; select 2", "invalid_statement_count"),
        ("delete from source", "write_statement_rejected"),
        ("create table copy as select * from source", "write_statement_rejected"),
        ("explain analyze select * from source", "write_statement_rejected"),
        ("set threads = 8", "write_statement_rejected"),
        ("install spatial", "write_statement_rejected"),
    ],
)
async def test_artifact_query_rejects_non_read_only_or_multiple_statements(
    sql: str,
    error_kind: str,
) -> None:
    source = relation(
        [("id", TableValueType.INTEGER)],
        [{"id": 1}],
    )

    with pytest.raises(
        ArtifactQueryExecutorError,
        match=rf"statement index 0.*\({error_kind}\)",
    ):
        await IsolatedDuckDbArtifactTableExecutor().execute(
            [SqlStatement(sql=sql)],
            ["source"],
            [source],
        )


async def test_artifact_query_cannot_read_host_files_or_change_configuration() -> None:
    source = relation(
        [("id", TableValueType.INTEGER)],
        [{"id": 1}],
    )
    executor = IsolatedDuckDbArtifactTableExecutor()

    with pytest.raises(
        ArtifactQueryExecutorError,
        match="Permission Error.*execution_error",
    ) as captured:
        await executor.execute(
            [SqlStatement(sql="select * from read_text('/etc/passwd')")],
            ["source"],
            [source],
        )

    assert "/etc/passwd" not in str(captured.value)
    settings = await executor.execute(
        [
            SqlStatement(
                sql=(
                    "select current_setting('enable_external_access') as external, "
                    "current_setting('lock_configuration') as locked"
                )
            )
        ],
        ["source"],
        [source],
    )
    assert settings[0].rows == [{"column_1": False, "column_2": True}]


async def test_artifact_query_timeout_terminates_the_one_shot_worker() -> None:
    source = relation(
        [("id", TableValueType.INTEGER)],
        [{"id": 1}],
    )
    executor = IsolatedDuckDbArtifactTableExecutor(wall_time_seconds=0.05)

    with pytest.raises(
        ArtifactQueryExecutorError,
        match="wall-time limit",
    ):
        await asyncio.wait_for(
            executor.execute(
                [
                    SqlStatement(
                        sql=("select sum(i) from range(1000000000000) values_table(i)")
                    )
                ],
                ["source"],
                [source],
            ),
            timeout=2,
        )
