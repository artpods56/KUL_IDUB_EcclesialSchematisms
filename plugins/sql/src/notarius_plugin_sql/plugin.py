from notarius_core.artifacts import Artifact
from notarius_core.runtime.persistence import InlineModelOutputWriter
from notarius_core.runtime.resolvers import InlineModelResolver

from notarius_plugin_sql import nodes
from notarius_plugin_sql.artifacts import SQL_RESULT, SQL_STATEMENT
from notarius_plugin_sql.declaration import SQL
from notarius_plugin_sql.models import SqlResult, SqlStatement


_NODE_MODULES = (nodes,)

SQL.register(
    Artifact(
        spec=SQL_STATEMENT,
        resolver=lambda context: InlineModelResolver(
            source=SQL_STATEMENT.key,
            target=SqlStatement,
            uow=context.uow,
        ),
        writer=lambda context: InlineModelOutputWriter(
            artifact_type=SQL_STATEMENT.key,
            model=SqlStatement,
            uow=context.uow,
        ),
    )
)
SQL.register(
    Artifact(
        spec=SQL_RESULT,
        resolver=lambda context: InlineModelResolver(
            source=SQL_RESULT.key,
            target=SqlResult,
            uow=context.uow,
        ),
        writer=lambda context: InlineModelOutputWriter(
            artifact_type=SQL_RESULT.key,
            model=SqlResult,
            uow=context.uow,
        ),
    )
)

__all__ = ["SQL"]
