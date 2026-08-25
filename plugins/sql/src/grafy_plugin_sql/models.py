from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from grafy_core.table_contracts import Table


type SqlValue = str | int | float | bool | None | list[SqlValue] | dict[str, SqlValue]


class SqlStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sql: StrictStr = Field(
        min_length=1,
        max_length=1_000_000,
        description=(
            "SQL statement using canonical named :parameter placeholders. "
            "Executors translate this binding syntax for their database driver."
        ),
    )
    parameters: dict[str, SqlValue] = Field(
        default_factory=dict,
        max_length=10_000,
        description=(
            "Named non-secret values bound to :name parameters. Values are "
            "persisted in the statement artifact."
        ),
    )


class SqlResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement_index: StrictInt = Field(ge=0)
    returns_rows: bool
    table: Table
    affected_rows: StrictInt | None = Field(default=None, ge=0)
