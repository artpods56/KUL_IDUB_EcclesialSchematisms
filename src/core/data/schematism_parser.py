import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any

import geopandas as gpd
import pandas as pd
from PIL import Image
from pydantic import BaseModel, Field
from shapely import wkt


logger = logging.getLogger(__name__)


class SchematismEntry(BaseModel):
    deanery: Optional[str] = None
    parish: Optional[str] = None
    dedication: Optional[str] = None
    building_material: Optional[str] = None


class SchematismPage(BaseModel):
    page_number: Optional[str] = None
    entries: List[SchematismEntry] = Field(default_factory=list)


class ShapefileGeneratorConfig(BaseModel):
    """
    Configuration for ShapefileGenerator.

    Fields are intentionally explicit so this object can be created from a config file
    or dependency-injected in tests.
    """

    csv_path: Path
    schematisms_dir: Path
    relative_shapefile_path: Path = Path("matryca/matryca.shp")
    schematism_name_column: str = "skany"
    file_name_column: str = "location"
    wkt_column: str = "the_geom"
    page_number_column: str = "strona_p"
    # mapping from schematism CSV columns to our entry fields
    column_mapping: Dict[str, str] = Field(
        default_factory=lambda: {
            "deanery": "dekanat",
            "parish": "parafia",
            "dedication": "wezwanie",
            "building_material": "material_typ",
        }
    )
    # optional subdirectory under schematism_dir where images live; if None, schematism_dir/<schematism_dir.stem>/ is used
    image_subdir: Optional[Path] = None


class ShapefileGenerator:
    def __init__(self, config: ShapefileGeneratorConfig) -> None:
        self._config = config
        self._cache: dict[str, gpd.GeoDataFrame] = {}  # cache per schematism

    def _load(self, schematism_name: str) -> gpd.GeoDataFrame:
        """Load and join data for a given schematism (cached)."""
        if schematism_name not in self._cache:
            df = self._load_csv_data(self._config.csv_path, schematism_name)
            joined = self._perform_spatial_join(df, schematism_name)
            self._cache[schematism_name] = joined
        return self._cache[schematism_name]

    def get_page_data(self, filename: str, schematism_name: str) -> SchematismPage:
        joined = self._load(schematism_name)
        rows = joined[joined[self._config.file_name_column] == filename]
        if rows.empty:
            return SchematismPage(page_number=None, entries=[])

        records = rows.to_dict(orient="records")
        page_number = records[0].get(self._config.page_number_column)

        m = self._config.column_mapping
        entries = [
            SchematismEntry(
                deanery=r.get(m["deanery"]),
                parish=r.get(m["parish"]),
                dedication=r.get(m["dedication"]),
                building_material=r.get(m["building_material"]),
            )
            for r in records
        ]
        return SchematismPage(page_number=page_number, entries=entries)

    def iter_pages(self, schematism_name: str) -> Iterator[dict[str, Any]]:
        """Yield samples for a given schematism name."""
        image_dir = self._resolve_image_dir(schematism_name)
        if not image_dir.exists():
            logger.warning("Image dir missing: %s", image_dir)
            return

        for image_path in sorted(image_dir.glob("*.jpg")):
            with Image.open(image_path) as image:
                page_data = self.get_page_data(image_path.name, schematism_name)
                yield {
                    "image": image,
                    "source": {"entries": [], "page_number": None},
                    "parsed": page_data.model_dump(),
                    "schematism_name": schematism_name,
                    "filename": image_path.name,
                }

    # ---- helpers updated ----

    def _resolve_image_dir(self, schematism_name: str) -> Path:
        if self._config.image_subdir is not None:
            return self._config.schematisms_dir / self._config.image_subdir
        return self._config.schematisms_dir / schematism_name

    def _load_csv_data(self, csv_path: Path, schematism_name: str) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[df[self._config.schematism_name_column] == schematism_name]
        if self._config.wkt_column not in df.columns:
            raise KeyError(f"Missing WKT column {self._config.wkt_column}")
        df = df.dropna(subset=[self._config.wkt_column]).copy()
        df["obj_geom"] = df[self._config.wkt_column].apply(wkt.loads)  # type: ignore
        return df

    def _construct_shapefile_path(self, schematism_name: str) -> Path:
        """Return resolved path to the shapefile defined in config."""
        return (
            self._config.schematisms_dir
            / schematism_name
            / self._config.relative_shapefile_path
        )

    def _perform_spatial_join(
        self, df: pd.DataFrame, schematism_name: str
    ) -> gpd.GeoDataFrame:
        """Perform spatial join between object geometries (from CSV) and page polygons (from shapefile)."""
        shapefile_path = self._construct_shapefile_path(schematism_name)
        if not shapefile_path.exists():
            raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")

        shp = gpd.read_file(shapefile_path)
        # keep the page geometry under an explicit name to avoid surprises
        shp = shp.rename_geometry("page_geom")
        entries = gpd.GeoDataFrame(df, geometry="obj_geom", crs=shp.crs)
        joined = gpd.sjoin(entries, shp, how="inner", predicate="intersects")
        return joined
