from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import registry

from notarius_core.artifacts import ArtifactObject
from notarius_core.domain.invocation_cache import InvocationCacheEntry
from notarius_core.domain.materialized_outputs import MaterializedNodeOutputs
from notarius_core.domain.node_secrets import EncryptedNodeSecret
from notarius_core.domain.saved_graphs import SavedGraph, SavedGraphDocument

from notarius_persistence import schema


mapper_registry = registry(metadata=schema.metadata)
metadata = schema.metadata


@dataclass
class SavedGraphRevisionRecord:
    graph_id: UUID
    revision: int
    name: str
    document: SavedGraphDocument
    created_at: datetime


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
        SavedGraphRevisionRecord,
        schema.saved_graph_revisions,
    )
    mapper_registry.map_imperatively(
        ArtifactObject,
        schema.artifact_objects,
    )
    mapper_registry.map_imperatively(
        InvocationCacheEntry,
        schema.invocation_cache_entries,
    )
    mapper_registry.map_imperatively(
        MaterializedNodeOutputs,
        schema.materialized_node_outputs,
    )
    mapper_registry.map_imperatively(
        EncryptedNodeSecret,
        schema.node_secrets,
    )
