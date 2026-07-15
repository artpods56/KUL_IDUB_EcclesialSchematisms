from sqlalchemy.orm import registry

from notarius_core.artifacts import ArtifactObject
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.saved_graphs import SavedGraph

from notarius_persistence import schema


mapper_registry = registry(metadata=schema.metadata)
metadata = schema.metadata


def start_mappers() -> None:
    if mapper_registry.mappers:
        return

    mapper_registry.map_imperatively(
        SavedGraph,
        schema.saved_graphs,
        version_id_col=schema.saved_graphs.c.revision,
        version_id_generator=False,
    )
    mapper_registry.map_imperatively(
        ArtifactObject,
        schema.artifact_objects,
    )
    mapper_registry.map_imperatively(
        MaterializedNodeOutputs,
        schema.materialized_node_outputs,
    )
