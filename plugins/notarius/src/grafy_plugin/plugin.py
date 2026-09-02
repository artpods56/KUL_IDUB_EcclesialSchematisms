from grafy_core.artifacts import Artifact
from grafy_core.runtime.persistence import InlineModelOutputWriter
from grafy_core.runtime.resolvers import InlineModelResolver

from grafy_plugin import nodes
from grafy_plugin.artifacts import (
    STRUCTURED_EXTRACTION_DATASET,
    StructuredExtractionDataset,
)
from grafy_plugin.declaration import PLUGIN


_NODE_MODULES = (nodes,)


PLUGIN.register(
    Artifact(
        spec=STRUCTURED_EXTRACTION_DATASET,
        resolver=lambda context: InlineModelResolver(
            source=STRUCTURED_EXTRACTION_DATASET.key,
            target=StructuredExtractionDataset,
            uow=context.uow,
        ),
        writer=lambda context: InlineModelOutputWriter(
            artifact_type=STRUCTURED_EXTRACTION_DATASET.key,
            model=StructuredExtractionDataset,
            uow=context.uow,
        ),
    )
)


__all__ = ["PLUGIN"]
