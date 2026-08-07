from notarius_core.artifacts import (
    ArtifactFieldProjection,
    ArtifactTypeKey,
    ArtifactTypeSpec,
)
from notarius_core.operators.tables import TABLE_DATA

from notarius_plugin_sql.models import SqlResult, SqlStatement


SQL_STATEMENT = ArtifactTypeSpec(
    key=ArtifactTypeKey("sql.statement", 1),
    title="SQL statement",
    payload_schema=SqlStatement.model_json_schema(),
)

SQL_RESULT = ArtifactTypeSpec(
    key=ArtifactTypeKey("sql.result", 1),
    title="SQL result",
    payload_schema=SqlResult.model_json_schema(),
    field_projections=(
        ArtifactFieldProjection(
            path=("table",),
            target=TABLE_DATA.key,
            title="Table",
        ),
    ),
)
