from pathlib import Path
from typing import Annotated, final, override

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from notarius_core.artifacts import NoConfig, NodeConfig, NodeInput, NodeOutput
from notarius_core.nodes import InPort, Node, NodeExecutionContext, OutPort
from notarius_core.plugins import NodeCachePolicy

from notarius_plugin_gis.artifacts import GEO_FEATURE_COLLECTION, GEO_MAP_DOCUMENT
from notarius_plugin_gis.declaration import GIS
from notarius_plugin_gis.models import (
    GeoFeatureCollection,
    GeoMapDocument,
    GeoMapLayer,
    combined_bounds,
)


class GeoJsonUploadError(RuntimeError):
    pass


class GeoJsonUploadItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    upload_key: StrictStr = Field(min_length=1)
    filename: StrictStr = Field(min_length=1)
    byte_size: StrictInt = Field(ge=0)


class GeoJsonUploadConfig(NodeConfig):
    uploads: list[GeoJsonUploadItem] = Field(
        min_length=1,
        max_length=1,
        description="One staged GeoJSON upload.",
    )


class GeoJsonUploadInput(NodeInput):
    pass


class GeoJsonUploadOutput(NodeOutput):
    features: Annotated[
        GeoFeatureCollection,
        OutPort(GEO_FEATURE_COLLECTION),
        Field(description="Validated WGS84 GeoJSON FeatureCollection."),
    ]


@GIS.node(
    operator_id="gis.geojson.upload",
    version=1,
    title="Import GeoJSON",
    factory=lambda context: ImportGeoJsonNode(uploads_dir=context.uploads_dir),
)
@final
class ImportGeoJsonNode(
    Node[GeoJsonUploadConfig, GeoJsonUploadInput, GeoJsonUploadOutput]
):
    """Imports one staged WGS84 GeoJSON FeatureCollection."""

    def __init__(self, uploads_dir: Path) -> None:
        self._uploads_dir = uploads_dir.expanduser().resolve()

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        config: GeoJsonUploadConfig,
        _inputs: GeoJsonUploadInput,
        /,
    ) -> GeoJsonUploadOutput:
        upload = config.uploads[0]
        relative_path = Path(upload.upload_key)
        if (
            relative_path.is_absolute()
            or relative_path.parts != (upload.upload_key,)
            or upload.upload_key in {".", ".."}
            or "\\" in upload.upload_key
        ):
            raise GeoJsonUploadError(
                f"GeoJSON upload key {upload.upload_key!r} must be one opaque relative name"
            )
        path = (self._uploads_dir / relative_path).resolve()
        if path.parent != self._uploads_dir:
            raise GeoJsonUploadError(
                f"GeoJSON upload key {upload.upload_key!r} resolves outside {self._uploads_dir}"
            )
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise GeoJsonUploadError(
                f"Failed to read staged GeoJSON upload {upload.upload_key!r} from {path}"
            ) from exc
        if len(content) != upload.byte_size:
            raise GeoJsonUploadError(
                f"Staged GeoJSON upload {upload.upload_key!r} changed size: expected "
                f"{upload.byte_size}, got {len(content)}"
            )
        try:
            features = GeoFeatureCollection.from_geojson_bytes(content, upload.filename)
        except ValueError as exc:
            raise GeoJsonUploadError(str(exc)) from exc
        return GeoJsonUploadOutput(features=features)


class ComposeMapInput(NodeInput):
    feature_collections: Annotated[
        list[GeoFeatureCollection],
        InPort(GEO_FEATURE_COLLECTION),
        Field(min_length=1, description="Ordered GeoJSON layers to display."),
    ]


class ComposeMapOutput(NodeOutput):
    map: Annotated[
        GeoMapDocument,
        OutPort(GEO_MAP_DOCUMENT),
        Field(description="Interactive map composition."),
    ]


_LAYER_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#ca8a04",
    "#db2777",
)


@GIS.node(
    operator_id="gis.map.compose",
    version=1,
    title="Compose map",
    cache_policy=NodeCachePolicy.EXACT,
)
@final
class ComposeMapNode(Node[NoConfig, ComposeMapInput, ComposeMapOutput]):
    """Composes an ordered feature-collection sequence into a map document."""

    @override
    async def run(
        self,
        _context: NodeExecutionContext,
        _config: NoConfig,
        inputs: ComposeMapInput,
        /,
    ) -> ComposeMapOutput:
        layers = [
            GeoMapLayer(
                id=f"layer-{index + 1}",
                title=collection.source_name,
                color=_LAYER_COLORS[index % len(_LAYER_COLORS)],
                feature_collection=collection,
            )
            for index, collection in enumerate(inputs.feature_collections)
        ]
        return ComposeMapOutput(
            map=GeoMapDocument(
                layers=layers,
                bounds=combined_bounds(inputs.feature_collections),
            )
        )
