from sqlalchemy.orm import registry

from notarius_core.domain.models import (
    Job,
    JobItem,
    OutputSchema,
    Project,
    Recipe,
    Source,
    SourceItem,
)
from notarius_persistence import schema

mapper_registry = registry(metadata=schema.metadata)
metadata = schema.metadata


def start_mappers() -> None:
    if mapper_registry.mappers:
        return
    mapper_registry.map_imperatively(Project, schema.projects)
    mapper_registry.map_imperatively(Source, schema.sources)
    mapper_registry.map_imperatively(SourceItem, schema.source_items)
    mapper_registry.map_imperatively(OutputSchema, schema.output_schemas)
    mapper_registry.map_imperatively(Recipe, schema.recipes)
    mapper_registry.map_imperatively(Job, schema.jobs)
    mapper_registry.map_imperatively(JobItem, schema.job_items)

