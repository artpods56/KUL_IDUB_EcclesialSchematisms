import json
from pathlib import Path

import pytest

from notarius_core.artifacts import InMemoryUnitOfWork, NoConfig
from notarius_core.nodes import NodeExecutionContext
from notarius_core.plugins import PluginOrigin, PluginRegistry
from notarius_core.runtime.materialization import MaterializationProvenance
from notarius_core.runtime.persistence import ArtifactWriteContext
from notarius_storage import LocalFileObjectStore
from notarius_plugin_gis.artifacts import GEO_FEATURE_COLLECTION, GEO_MAP_DOCUMENT
from notarius_plugin_gis.models import GeoFeatureCollection
from notarius_plugin_gis.nodes import (
    ComposeMapInput,
    ComposeMapNode,
    GeoJsonUploadConfig,
    GeoJsonUploadError,
    GeoJsonUploadInput,
    GeoJsonUploadItem,
    ImportGeoJsonNode,
)
from notarius_plugin_gis.plugin import GIS
from notarius_plugin_gis.persistence import SpatialJsonOutputWriter, SpatialJsonResolver


def feature_collection(*coordinates: tuple[float, float]) -> bytes:
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": f"place-{index}"},
                    "geometry": {"type": "Point", "coordinates": list(position)},
                }
                for index, position in enumerate(coordinates)
            ],
        }
    ).encode("utf-8")


async def import_upload(
    uploads_dir: Path,
    *,
    filename: str,
    content: bytes,
) -> GeoFeatureCollection:
    uploads_dir.mkdir(parents=True, exist_ok=True)
    upload_key = f"staged-{filename}"
    (uploads_dir / upload_key).write_bytes(content)
    result = await ImportGeoJsonNode(uploads_dir).run(
        NodeExecutionContext(node_id="import"),
        GeoJsonUploadConfig(
            uploads=[
                GeoJsonUploadItem(
                    upload_key=upload_key,
                    filename=filename,
                    byte_size=len(content),
                )
            ]
        ),
        GeoJsonUploadInput(),
    )
    return result.features


def test_gis_is_an_external_plugin_with_collect_compatible_artifacts() -> None:
    registry = PluginRegistry()
    registry.install(GIS, origin=PluginOrigin.EXTERNAL)
    registry.freeze()

    assert GIS.slug == "external.gis"
    assert {artifact.key for artifact in GIS.artifact_types} == {
        GEO_FEATURE_COLLECTION.key,
        GEO_MAP_DOCUMENT.key,
    }
    assert registry.node_registration("gis.geojson.upload", 1).node_class.output_contract.ports[
        "features"
    ].produces == GEO_FEATURE_COLLECTION.key
    compose_input = registry.node_registration(
        "gis.map.compose", 1
    ).node_class.input_contract.ports["feature_collections"]
    assert compose_input.accepts == GEO_FEATURE_COLLECTION.key
    assert compose_input.shape.value == "many"


@pytest.mark.asyncio
async def test_imports_wgs84_geojson_and_composes_ordered_layers(tmp_path: Path) -> None:
    first = await import_upload(
        tmp_path / "uploads",
        filename="cities.geojson",
        content=feature_collection((13.405, 52.52), (2.3522, 48.8566)),
    )
    second = await import_upload(
        tmp_path / "uploads",
        filename="offices.geojson",
        content=feature_collection((-0.1276, 51.5072)),
    )

    result = await ComposeMapNode().run(
        NodeExecutionContext(node_id="map"),
        NoConfig(),
        ComposeMapInput(feature_collections=[first, second]),
    )

    assert [layer.title for layer in result.map.layers] == [
        "cities.geojson",
        "offices.geojson",
    ]
    assert [layer.color for layer in result.map.layers] == ["#2563eb", "#dc2626"]
    assert result.map.bounds == (-0.1276, 48.8566, 13.405, 52.52)


@pytest.mark.asyncio
async def test_import_rejects_coordinates_outside_wgs84(tmp_path: Path) -> None:
    with pytest.raises(GeoJsonUploadError, match="outside WGS84"):
        await import_upload(
            tmp_path / "uploads",
            filename="projected.geojson",
            content=feature_collection((4_000_000, 5_000_000)),
        )


@pytest.mark.asyncio
async def test_import_rejects_non_feature_collection(tmp_path: Path) -> None:
    with pytest.raises(GeoJsonUploadError, match="not a valid GeoJSON FeatureCollection"):
        await import_upload(
            tmp_path / "uploads",
            filename="point.geojson",
            content=b'{"type":"Point","coordinates":[13.4,52.5]}',
        )


@pytest.mark.asyncio
async def test_feature_collection_storage_round_trip(tmp_path: Path) -> None:
    collection = GeoFeatureCollection.from_geojson_bytes(
        feature_collection((13.405, 52.52)),
        "cities.geojson",
    )
    unit_of_work = InMemoryUnitOfWork()
    storage = LocalFileObjectStore(tmp_path / "objects")
    writer = SpatialJsonOutputWriter(
        artifact_type=GEO_FEATURE_COLLECTION.key,
        model=GeoFeatureCollection,
        content_type="application/geo+json",
        storage=storage,
        uow=unit_of_work,
        bucket="test",
        storage_backend="local",
    )
    ref = await writer.write(
        collection,
        ArtifactWriteContext(
            node_context=NodeExecutionContext(node_id="import"),
            provenance=MaterializationProvenance(refs_by_input={}),
        ),
    )
    resolver = SpatialJsonResolver(
        source=GEO_FEATURE_COLLECTION.key,
        target=GeoFeatureCollection,
        uow=unit_of_work,
        storage=storage,
    )

    assert await resolver.resolve(ref) == collection
    async with unit_of_work as uow:
        artifact = await uow.artifacts.get(ref.artifact_id)
    assert artifact is not None
    assert artifact.inline_payload is None
    assert artifact.content_type == "application/geo+json"
