import base64
import json
import re
import sys
from collections.abc import Sequence
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import cast
from uuid import UUID

import duckdb
from pydantic import ValidationError

from grafy_core.table_contracts import (
    Table,
    TableColumn,
    TableValue,
    TableValueType,
)

from grafy_plugin_sql.artifact_query import (
    ArtifactQueryWorkerError,
    ArtifactQueryWorkerFailure,
    ArtifactQueryWorkerRequest,
    ArtifactQueryWorkerSuccess,
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    MAX_RESULT_ROWS,
)


MAX_INPUT_ROWS = 1_000_000
MAX_INPUT_CELLS = 10_000_000
PARAMETER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DOLLAR_QUOTE_START_PATTERN = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


class ArtifactQueryRejected(RuntimeError):
    def __init__(
        self,
        kind: str,
        message: str,
        *,
        statement_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.statement_index = statement_index


def _quote_identifier(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _duckdb_type(column: TableColumn, rows: list[dict[str, TableValue]]) -> str:
    if column.value_type is TableValueType.TEXT:
        return "VARCHAR"
    if column.value_type is TableValueType.INTEGER:
        return "BIGINT"
    if column.value_type is TableValueType.NUMBER:
        return "DOUBLE"
    if column.value_type is TableValueType.BOOLEAN:
        return "BOOLEAN"
    if column.value_type is TableValueType.DATE:
        return "DATE"
    if column.value_type is TableValueType.DATETIME:
        parsed_datetimes = [
            datetime.fromisoformat(cast(str, row[column.id]))
            for row in rows
            if row[column.id] is not None
        ]
        if any(value.utcoffset() is not None for value in parsed_datetimes):
            return "TIMESTAMP WITH TIME ZONE"
        return "TIMESTAMP"
    if column.value_type is TableValueType.BINARY:
        return "BLOB"
    if column.value_type in {
        TableValueType.JSON,
        TableValueType.MIXED,
        TableValueType.UNKNOWN,
    }:
        return "JSON"
    if column.value_type is TableValueType.DECIMAL:
        decimals: list[Decimal] = []
        for row in rows:
            value = row[column.id]
            if value is not None:
                decimals.append(Decimal(cast(str, value)))
        if not decimals:
            return "DECIMAL(38, 18)"
        integer_digits = 0
        scale = 0
        for value in decimals:
            if not value.is_finite():
                raise ArtifactQueryRejected(
                    "unsupported_input",
                    f"Decimal column {column.id!r} contains a non-finite value",
                )
            _sign, digits, exponent = value.as_tuple()
            if not isinstance(exponent, int):
                raise ArtifactQueryRejected(
                    "unsupported_input",
                    f"Decimal column {column.id!r} contains an invalid exponent",
                )
            value_scale = max(-exponent, 0)
            value_integer_digits = max(len(digits) + exponent, 0)
            integer_digits = max(integer_digits, value_integer_digits)
            scale = max(scale, value_scale)
        precision = max(1, integer_digits + scale)
        if precision > 38:
            raise ArtifactQueryRejected(
                "unsupported_input",
                f"Decimal column {column.id!r} requires precision {precision}; "
                "DuckDB supports at most 38 digits",
            )
        return f"DECIMAL({precision}, {scale})"
    raise ArtifactQueryRejected(
        "unsupported_input",
        f"Column {column.id!r} has unsupported type {column.value_type.value!r}",
    )


def _input_value(value: TableValue, value_type: TableValueType) -> object:
    if value is None:
        return None
    if value_type is TableValueType.DECIMAL:
        try:
            return Decimal(cast(str, value))
        except InvalidOperation as exc:
            raise ArtifactQueryRejected(
                "invalid_input",
                "A decimal table value could not be decoded",
            ) from exc
    if value_type is TableValueType.DATE:
        return date.fromisoformat(cast(str, value))
    if value_type is TableValueType.DATETIME:
        return datetime.fromisoformat(cast(str, value))
    if value_type is TableValueType.BINARY:
        return base64.b64decode(cast(str, value), validate=True)
    if value_type in {
        TableValueType.JSON,
        TableValueType.MIXED,
        TableValueType.UNKNOWN,
    }:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return value


def _load_relation(
    connection: duckdb.DuckDBPyConnection,
    alias: str,
    table: Table,
) -> None:
    if not table.columns:
        raise ArtifactQueryRejected(
            "unsupported_input",
            f"Relation {alias!r} has no columns",
        )
    definitions = ", ".join(
        f"{_quote_identifier(column.id)} {_duckdb_type(column, table.rows)}"
        for column in table.columns
    )
    try:
        connection.execute(f"CREATE TABLE {_quote_identifier(alias)} ({definitions})")
        if table.rows:
            placeholders = ", ".join("?" for _column in table.columns)
            connection.executemany(
                f"INSERT INTO {_quote_identifier(alias)} VALUES ({placeholders})",
                [
                    [
                        _input_value(row[column.id], column.value_type)
                        for column in table.columns
                    ]
                    for row in table.rows
                ],
            )
    except ArtifactQueryRejected:
        raise
    except Exception as exc:
        raise ArtifactQueryRejected(
            "invalid_input",
            f"Relation {alias!r} could not be loaded: {_engine_message(exc)}",
        ) from exc


def _translate_named_parameters(sql: str) -> str:
    translated: list[str] = []
    index = 0
    while index < len(sql):
        character = sql[index]
        if sql.startswith("--", index):
            end = sql.find("\n", index + 2)
            if end == -1:
                translated.append(sql[index:])
                break
            translated.append(sql[index:end])
            index = end
            continue
        if sql.startswith("/*", index):
            start = index
            depth = 1
            index += 2
            while index < len(sql) and depth > 0:
                if sql.startswith("/*", index):
                    depth += 1
                    index += 2
                elif sql.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            translated.append(sql[start:index])
            continue
        if character in {"'", '"'}:
            quote = character
            start = index
            index += 1
            while index < len(sql):
                if sql[index] == quote:
                    index += 1
                    if index < len(sql) and sql[index] == quote:
                        index += 1
                        continue
                    break
                index += 1
            translated.append(sql[start:index])
            continue
        if character == "$":
            match = DOLLAR_QUOTE_START_PATTERN.match(sql, index)
            if match is not None:
                delimiter = match.group(0)
                end = sql.find(delimiter, match.end())
                if end != -1:
                    end += len(delimiter)
                    translated.append(sql[index:end])
                    index = end
                    continue
        if (
            character == ":"
            and index + 1 < len(sql)
            and (sql[index + 1].isalpha() or sql[index + 1] == "_")
            and (index == 0 or sql[index - 1] != ":")
        ):
            end = index + 2
            while end < len(sql) and (sql[end].isalnum() or sql[end] == "_"):
                end += 1
            translated.append("$" + sql[index + 1 : end])
            index = end
            continue
        translated.append(character)
        index += 1
    return "".join(translated)


def _engine_message(exc: Exception) -> str:
    first_line = str(exc).splitlines()[0].strip()
    if not first_line:
        return type(exc).__name__
    redacted = re.sub(r'"(?:[^"]|"")*"', '"<redacted>"', first_line)
    redacted = re.sub(r"'(?:[^']|'')*'", "'<redacted>'", redacted)
    return redacted[:500]


def _table_value(value: object, duckdb_type: str) -> TableValue:
    if value is None:
        return None
    if duckdb_type == "JSON" and isinstance(value, str):
        decoded = json.loads(value)
        return _table_value(decoded, "")
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date | time | UUID):
        return value.isoformat() if not isinstance(value, UUID) else str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, list | tuple):
        return [_table_value(item, "") for item in cast(Sequence[object], value)]
    if isinstance(value, dict):
        return {
            str(key): _table_value(item, "")
            for key, item in cast(dict[object, object], value).items()
        }
    raise ArtifactQueryRejected(
        "unsupported_output",
        f"DuckDB returned unsupported value type {type(value).__name__}",
    )


def _value_type(value: TableValue) -> TableValueType:
    if isinstance(value, bool):
        return TableValueType.BOOLEAN
    if isinstance(value, int):
        return TableValueType.INTEGER
    if isinstance(value, float):
        return TableValueType.NUMBER
    if isinstance(value, list | dict):
        return TableValueType.JSON
    if isinstance(value, str):
        return TableValueType.TEXT
    return TableValueType.UNKNOWN


def _column_value_type(
    values: Sequence[TableValue],
    duckdb_type: str,
) -> TableValueType:
    if duckdb_type.startswith("DECIMAL"):
        return TableValueType.DECIMAL
    if duckdb_type == "DATE":
        return TableValueType.DATE
    if duckdb_type.startswith("TIMESTAMP"):
        return TableValueType.DATETIME
    if duckdb_type == "BLOB":
        return TableValueType.BINARY
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


def _execute_statement(
    connection: duckdb.DuckDBPyConnection,
    sql: str,
    parameters: dict[str, object],
    statement_index: int,
) -> Table:
    for name in parameters:
        if PARAMETER_NAME_PATTERN.fullmatch(name) is None:
            raise ArtifactQueryRejected(
                "invalid_parameters",
                f"Parameter name {name!r} is not a portable SQL identifier",
                statement_index=statement_index,
            )
    translated_sql = _translate_named_parameters(sql)
    try:
        statements = connection.extract_statements(translated_sql)
    except Exception as exc:
        raise ArtifactQueryRejected(
            "invalid_statement",
            f"Query could not be parsed: {_engine_message(exc)}",
            statement_index=statement_index,
        ) from exc
    if len(statements) != 1:
        raise ArtifactQueryRejected(
            "invalid_statement_count",
            "Each statement artifact must contain exactly one SQL statement",
            statement_index=statement_index,
        )
    prepared = statements[0]
    if prepared.type != duckdb.StatementType.SELECT:
        raise ArtifactQueryRejected(
            "write_statement_rejected",
            "Each statement artifact must contain one read-only query",
            statement_index=statement_index,
        )
    expected_parameters = set(prepared.named_parameters)
    supplied_parameters = set(parameters)
    if expected_parameters != supplied_parameters:
        missing = sorted(expected_parameters - supplied_parameters)
        unexpected = sorted(supplied_parameters - expected_parameters)
        details: list[str] = []
        if missing:
            details.append(f"missing {missing!r}")
        if unexpected:
            details.append(f"unexpected {unexpected!r}")
        raise ArtifactQueryRejected(
            "invalid_parameters",
            "Bound parameters do not match query placeholders: " + ", ".join(details),
            statement_index=statement_index,
        )
    try:
        result = connection.execute(prepared.query, parameters)
        rows = result.fetchmany(MAX_RESULT_ROWS + 1)
        if len(rows) > MAX_RESULT_ROWS:
            raise ArtifactQueryRejected(
                "result_row_limit",
                f"Query returned more than {MAX_RESULT_ROWS} rows",
                statement_index=statement_index,
            )
        description = result.description or []
        column_titles = [str(item[0]) for item in description]
        duckdb_types = [str(item[1]) for item in description]
        column_ids = [
            f"column_{column_index + 1}" for column_index in range(len(column_titles))
        ]
        table_rows = [
            {
                column_id: _table_value(value, duckdb_type)
                for column_id, value, duckdb_type in zip(
                    column_ids,
                    row,
                    duckdb_types,
                    strict=True,
                )
            }
            for row in rows
        ]
        return Table(
            columns=[
                TableColumn(
                    id=column_id,
                    title=title,
                    value_type=_column_value_type(
                        [row[column_id] for row in table_rows],
                        duckdb_type,
                    ),
                )
                for column_id, title, duckdb_type in zip(
                    column_ids,
                    column_titles,
                    duckdb_types,
                    strict=True,
                )
            ],
            rows=table_rows,
        )
    except ArtifactQueryRejected:
        raise
    except Exception as exc:
        raise ArtifactQueryRejected(
            "execution_error",
            f"DuckDB could not execute the query: {_engine_message(exc)}",
            statement_index=statement_index,
        ) from exc


def _execute_batch(request: ArtifactQueryWorkerRequest) -> list[Table]:
    total_rows = sum(len(relation.table.rows) for relation in request.relations)
    total_cells = sum(
        len(relation.table.rows) * len(relation.table.columns)
        for relation in request.relations
    )
    if total_rows > MAX_INPUT_ROWS:
        raise ArtifactQueryRejected(
            "input_row_limit",
            f"Relations contain {total_rows} rows, exceeding the "
            f"{MAX_INPUT_ROWS}-row limit",
        )
    if total_cells > MAX_INPUT_CELLS:
        raise ArtifactQueryRejected(
            "input_cell_limit",
            f"Relations contain {total_cells} cells, exceeding the "
            f"{MAX_INPUT_CELLS}-cell limit",
        )
    connection = duckdb.connect(
        ":memory:",
        config={
            "allow_community_extensions": "false",
            "allow_persistent_secrets": "false",
            "allow_unsigned_extensions": "false",
            "autoinstall_known_extensions": "false",
            "autoload_known_extensions": "false",
            "enable_external_access": "false",
            "enable_external_file_cache": "false",
            "max_temp_directory_size": "0GB",
            "memory_limit": "256MB",
            "preserve_insertion_order": "false",
            "threads": "1",
        },
    )
    statement_connections: list[duckdb.DuckDBPyConnection] = []
    try:
        connection.execute("SET TimeZone = 'UTC'")
        for relation in request.relations:
            _load_relation(connection, relation.alias, relation.table)
        for _statement in request.statements:
            statement_connection = connection.cursor()
            statement_connection.execute("SET TimeZone = 'UTC'")
            statement_connection.execute("SET disabled_filesystems = 'LocalFileSystem'")
            statement_connections.append(statement_connection)
        connection.execute("SET disabled_filesystems = 'LocalFileSystem'")
        connection.execute("SET lock_configuration = true")
        tables: list[Table] = []
        for index, (statement, statement_connection) in enumerate(
            zip(
                request.statements,
                statement_connections,
                strict=True,
            )
        ):
            tables.append(
                _execute_statement(
                    statement_connection,
                    statement.sql,
                    cast(dict[str, object], statement.parameters),
                    index,
                )
            )
        return tables
    finally:
        for statement_connection in statement_connections:
            statement_connection.close()
        connection.close()


def _write_response(
    response: ArtifactQueryWorkerSuccess | ArtifactQueryWorkerFailure,
) -> None:
    content = response.model_dump_json().encode("utf-8")
    if len(content) > MAX_OUTPUT_BYTES:
        content = (
            ArtifactQueryWorkerFailure(
                error=ArtifactQueryWorkerError(
                    kind="result_byte_limit",
                    message=(
                        f"Serialized query results exceed the {MAX_OUTPUT_BYTES}-byte "
                        "output limit"
                    ),
                )
            )
            .model_dump_json()
            .encode("utf-8")
        )
    sys.stdout.buffer.write(content)
    sys.stdout.buffer.flush()


def run_worker() -> int:
    content = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if len(content) > MAX_INPUT_BYTES:
        _write_response(
            ArtifactQueryWorkerFailure(
                error=ArtifactQueryWorkerError(
                    kind="input_byte_limit",
                    message=(f"Worker input exceeds the {MAX_INPUT_BYTES}-byte limit"),
                )
            )
        )
        return 0
    try:
        request = ArtifactQueryWorkerRequest.model_validate_json(content)
    except ValidationError:
        _write_response(
            ArtifactQueryWorkerFailure(
                error=ArtifactQueryWorkerError(
                    kind="invalid_request",
                    message="Worker input did not match the artifact query contract",
                )
            )
        )
        return 0
    try:
        tables = _execute_batch(request)
    except ArtifactQueryRejected as exc:
        _write_response(
            ArtifactQueryWorkerFailure(
                error=ArtifactQueryWorkerError(
                    kind=exc.kind,
                    message=str(exc),
                    statement_index=exc.statement_index,
                )
            )
        )
        return 0
    except Exception as exc:
        _write_response(
            ArtifactQueryWorkerFailure(
                error=ArtifactQueryWorkerError(
                    kind=type(exc).__name__,
                    message="Artifact query worker failed while preparing the batch",
                )
            )
        )
        return 0
    _write_response(ArtifactQueryWorkerSuccess(tables=tables))
    return 0


__all__ = ["ArtifactQueryRejected", "run_worker"]
