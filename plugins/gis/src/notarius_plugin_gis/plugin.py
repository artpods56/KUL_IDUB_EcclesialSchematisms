from typing import cast

from notarius_core.runtime.resolvers import Resolver

from notarius_plugin_gis import nodes
from notarius_plugin_gis.artifacts import GEO_FEATURE_COLLECTION, GEO_MAP_DOCUMENT
from notarius_plugin_gis.declaration import GIS
from notarius_plugin_gis.models import GeoFeatureCollection, GeoMapDocument
from notarius_plugin_gis.persistence import SpatialJsonOutputWriter, SpatialJsonResolver

_NODE_MODULES = (nodes,)


GIS.register_artifact_type(GEO_FEATURE_COLLECTION)
GIS.register_artifact_type(GEO_MAP_DOCUMENT)

GIS.register_writer(
    lambda context: SpatialJsonOutputWriter(
        artifact_type=GEO_FEATURE_COLLECTION.key,
        model=GeoFeatureCollection,
        content_type="application/geo+json",
        storage=context.storage,
        uow=context.uow,
        bucket=context.bucket,
        storage_backend=context.storage_backend,
    )
)
GIS.register_writer(
    lambda context: SpatialJsonOutputWriter(
        artifact_type=GEO_MAP_DOCUMENT.key,
        model=GeoMapDocument,
        content_type="application/json",
        storage=context.storage,
        uow=context.uow,
        bucket=context.bucket,
        storage_backend=context.storage_backend,
    )
)
GIS.register_resolver(
    lambda context: cast(
        Resolver[object],
        SpatialJsonResolver(
            source=GEO_FEATURE_COLLECTION.key,
            target=GeoFeatureCollection,
            uow=context.uow,
            storage=context.storage,
        ),
    )
)

__all__ = ["GIS"]
