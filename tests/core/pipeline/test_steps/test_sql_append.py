import pandas as pd
import pytest
import sqlite3

from core.pipeline.steps.export import AppendDataFrameToSQLStep


@pytest.fixture()
def sqlite_conn(tmp_path):
    db_path = tmp_path / "test.db"
    con = sqlite3.connect(db_path)
    try:
        yield con
    finally:
        con.close()


def test_append_to_sqlite_connection(sqlite_conn) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    step = AppendDataFrameToSQLStep(table_name="t_data", connection=sqlite_conn, if_exists="fail", index=False)

    # First write creates table
    out = step.process(df)
    assert out is df

    # Verify contents
    rows = pd.read_sql_query("SELECT * FROM t_data ORDER BY a ASC", sqlite_conn)
    assert list(rows.columns) == ["a", "b"]
    assert rows.shape == (2, 2)

    # Append mode
    step_append = AppendDataFrameToSQLStep(table_name="t_data", connection=sqlite_conn, if_exists="append", index=False)
    step_append.process(df)
    rows2 = pd.read_sql_query("SELECT COUNT(*) AS cnt FROM t_data", sqlite_conn)
    assert int(rows2.loc[0, "cnt"]) == 4


def test_invalid_connection_type_raises() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        AppendDataFrameToSQLStep(table_name="t", connection=object()).process(df)


