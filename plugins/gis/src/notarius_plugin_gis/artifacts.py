from notarius_core.artifacts import ArtifactTypeKey, ArtifactTypeSpec


GEO_FEATURE_COLLECTION = ArtifactTypeSpec(
    key=ArtifactTypeKey("geo.feature_collection", 1),
    title="GeoJSON feature collection",
)

GEO_MAP_DOCUMENT = ArtifactTypeSpec(
    key=ArtifactTypeKey("geo.map_document", 1),
    title="Map document",
)
