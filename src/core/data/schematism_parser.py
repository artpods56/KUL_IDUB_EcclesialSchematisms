import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from typing import Dict, List, Optional, Any
import os
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from typing import cast



from pydantic import BaseModel

class SchematismEntry(BaseModel):

    parish: Optional[str] = None
    dedication: Optional[str] = None
    building_material: Optional[str] = None
    
class SchematismPage(BaseModel):
    deanery: Optional[str] = None
    page_number: Optional[str] = None
    entries: List[SchematismEntry]

class SchematismParser:
    """
    A parser for schematism data that combines CSV entries with shapefile page locations.
    """
    
    def __init__(self, csv_path: str, schematism_name: str):
        """
        Initialize the SchematismParser.
        
        Args:
            csv_path: Path to the CSV file containing schematism data
            shapefile_path: Optional path to shapefile. If not provided, will be inferred.
        """

        load_dotenv()
        self.schematism_name = schematism_name
        schematisms_path = os.getenv("SCHEMATISMS_PATH", None)
        if schematisms_path is None:
            raise ValueError("SCHEMATISMS_PATH environment variable is not set")
        else:
            self.schematisms_path = Path(schematisms_path)
            self._load_shapefile(schematism_name)

        self.schematism_name = schematism_name
        self.csv_path = csv_path
        
        # Load the CSV data
        self._load_csv_data()
        self.joined = self._perform_spatial_join()
        
    def _load_csv_data(self):
        """Load and filter the CSV data."""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.df = self.df[self.df["skany"] == self.schematism_name]
            print(f"Loaded {len(self.df)} entries with facsimile data")
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            self.df = pd.DataFrame()
    
    def _load_shapefile(self, schematism_name: str):
        """Load shapefile data."""
        try:
            shapefile_path = self.schematisms_path / schematism_name / "matryca" / "matryca.shp"
            self.gdf_pages = gpd.read_file(shapefile_path)
            print(f"Loaded shapefile with {len(self.gdf_pages)} pages")
            return True
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            return False
    
    def _perform_spatial_join(self):
        """Attach page-level polygons to object polygons and return joined GeoDataFrame."""
        if self.gdf_pages is None:
            print("No shapefile loaded – cannot perform spatial join.")
            return None

        # ------------------------------------------------------------------
        # 1.  Prepare entries GeoDataFrame (objects).
        # ------------------------------------------------------------------
        # "the_geom" may contain NaNs → drop them first.
        self.df = self.df.dropna(subset=["the_geom"])  # type: ignore[arg-type]

        # Convert WKT string → shapely geometry
        self.df["obj_geom"] = self.df["the_geom"].apply(wkt.loads)

        gdf_entries = gpd.GeoDataFrame(self.df, geometry="obj_geom", crs=self.gdf_pages.crs)

        # ------------------------------------------------------------------
        # 2.  Ensure page polygons are available as "page_geom" column so that
        #     both geometries survive the spatial join. GeoPandas discards the
        #     *right-hand* geometry if its column name is still "geometry" –
        #     therefore we duplicate it under a safe name, set that as the
        #     active geometry and use this copy for the join.
        # ------------------------------------------------------------------
        # Duplicate the geometry into a *non-active* column so it survives
        # the join; keep the original "geometry" column active for the
        # spatial predicate.
        gdf_pages = self.gdf_pages.copy()
        gdf_pages["page_geom"] = gdf_pages.geometry

        # ------------------------------------------------------------------
        # 3.  Spatial join (works row-wise, keeps left geometry (obj_geom)
        #     and retains page attributes, *including page_geom*.
        # ------------------------------------------------------------------
        joined = gpd.sjoin(gdf_entries, gdf_pages, how="inner", predicate="intersects")

        print(f"Found {len(joined)} spatial matches for {self.schematism_name}")
        return joined
    
    def load_schematism(self, schematism_name: str, shapefile_path: Optional[str] = None):
        """
        Load a specific schematism and perform spatial matching.
        
        Args:
            schematism_name: Name of the schematism (e.g., 'wloclawek_1872')
            shapefile_path: Path to the corresponding shapefile
        """
        # Use provided shapefile path or try to infer it
        if shapefile_path:
            self.shapefile_path = shapefile_path
        elif not self.shapefile_path:
            # Try to infer shapefile path based on schematism name
            base_dir = "/Users/user/Projects/AI_Osrodek/data/schematyzmy"
            inferred_path = os.path.join(base_dir, schematism_name, "matryca", "matryca.shp")
            if os.path.exists(inferred_path):
                self.shapefile_path = inferred_path
            else:
                print(f"Could not find shapefile for {schematism_name}")
                return False
        
        # Load shapefile
        if not self._load_shapefile(self.shapefile_path):
            return False
        
        #
        
        return False


    def load_image(self, filename: str) -> Image.Image:
        """Load an image from the schematism directory."""
        img_path = self.schematisms_path / Path(self.schematism_name) / Path(filename)
        return Image.open(img_path)

    def obj_bbox_pixels(
        self,
        row: pd.Series,
    ) -> tuple[int, int, int, int]:
        """Convert *ARC-GIS* object geometry to image-pixel bounding box.

        Parameters
        ----------
        row : pd.Series
            A row obtained from *self.joined* – must contain "page_geom",
            "obj_geom" and "file_name" columns.
        image_dir : str | pathlib.Path
            Directory that contains the page image files.

        Returns
        -------
        (x1, y1, x2, y2) : tuple[int, int, int, int]
            Bounding box of the object in **pixel** coordinates.
        """

        # Extract geometries (Shapely polygons)
        page_geom = cast(BaseGeometry, row["page_geom"])
        obj_geom  = cast(BaseGeometry, row["obj_geom"])

        # Make sure they are really available – easier debugging later.
        if page_geom is None or obj_geom is None:
            raise ValueError("Row does not have required geometries (page_geom / obj_geom)")

        # Open image only to read its pixel dimensions.
        img_path = self.schematisms_path / Path(self.schematism_name) / Path(str(row["location"]))
        w_px, h_px = Image.open(img_path).size  # Pillow returns (width, height)

        minx_p, miny_p, maxx_p, maxy_p = page_geom.bounds
        minx_o, miny_o, maxx_o, maxy_o = obj_geom.bounds

        # Scale factors between world units and pixel units
        sx = w_px / (maxx_p - minx_p)
        sy = h_px / (maxy_p - miny_p)

        # Convert → pixel space; note that y axis in images is flipped 🤸‍♂️
        x1 = (minx_o - minx_p) * sx
        x2 = (maxx_o - minx_p) * sx
        y1 = (maxy_p - maxy_o) * sy  # invert y
        y2 = (maxy_p - miny_o) * sy

        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def parse_results(self, structured_results: List[Dict[str, Any]]) -> Dict[str, Any]:

        entries = []

        if len(structured_results) == 0:
            return SchematismPage(
                deanery=None,
                page_number=None,
                entries=[]
            ).model_dump()

        for result in structured_results:
            entries.append(SchematismEntry(
                parish=result["parafia"],
                dedication=result["wezwanie"],
                building_material=result["material_typ"]
            ))

        return SchematismPage(
            deanery=structured_results[0]["dekanat"],
            page_number=str(structured_results[0]["strona_p"]), 
            entries=entries).model_dump()
    
    def get_page_info(self, filename: str) -> Dict[str, Any]:
        """
        Get information about a specific page.
        
        Args:
            filename: The filename of the page (e.g., 'wloclawek_1872_0005.jpg')
            schematism_name: Optional schematism name. If not provided, will try to infer.
        
        Returns:
            Dictionary containing page information
        """
        
        structured_results = self.joined[self.joined["location"] == filename].to_dict(orient="records")
    

        return self.parse_results(structured_results)

    def get_available_schematisms(self) -> List[str]:
        """Return list of schematism names present in the *full CSV* (not just current join)."""
        if getattr(self, "df", None) is None or self.df.empty:
            return []

        return sorted(list(self.df["skany"].unique()))  # type: ignore[attr-defined]
    
    def get_schematism_stats(self) -> Dict[str, Any]:
        """Quick statistics for the *currently loaded* schematism."""

        if getattr(self, "joined", None) is None or self.joined is None:
            return {}

        joined_data = self.joined

        return {
            "total_entries": len(joined_data),
            "unique_pages": joined_data["location"].nunique() if "location" in joined_data.columns else 0,
            "deaneries": joined_data["dekanat"].unique().tolist() if "dekanat" in joined_data.columns else [],
            "dioceses": joined_data["diecezja"].unique().tolist() if "diecezja" in joined_data.columns else [],
        }
