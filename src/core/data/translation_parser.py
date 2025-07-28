import os
import json
from pathlib import Path

from thefuzz import fuzz, process

from structlog import get_logger
logger = get_logger(__name__)


from core.utils.shared import REPOSITORY_ROOT

class Parser:
    def __init__(self, building_material_mapping = None, dedication_mapping = None, fuzzy_threshold=80):

        building_material_mapping_path = os.getenv("BUILDING_MATERIAL_MAPPINGS", None)
        dedication_mapping_path = os.getenv("SAINTS_MAPPINGS", None)

        if not building_material_mapping_path or not dedication_mapping_path:
            raise ValueError("Set BUILDING_MATERIAL_MAPPINGS and SAINTS_MAPPINGS in .env file to point at mappings")

        if building_material_mapping is None:
            with open(REPOSITORY_ROOT / Path(building_material_mapping_path), "r") as f:
                building_material_mapping = json.load(f)

        if dedication_mapping is None:
            with open(REPOSITORY_ROOT / Path(dedication_mapping_path), "r") as f:
                dedication_mapping = json.load(f)

        if building_material_mapping_path:
            with open(REPOSITORY_ROOT / Path(building_material_mapping_path), "r") as f:
                building_material_mapping = json.load(f)
        else:
            building_material_mapping = building_material_mapping

        self.building_material_mapping = building_material_mapping
        self.dedication_mapping = dedication_mapping
        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_scorer = fuzz.ratio

    def fuzzy_match(self, text: str, keys: list[str]):
        return process.extractOne(text, keys, scorer=self.fuzzy_scorer, score_cutoff=self.fuzzy_threshold)


    def _parse_building_material(self, text: str):
        for key, value in self.building_material_mapping.items():
            if key == text:
                return value
        logger.debug(f"Fuzzy matching: {text}")
        match = self.fuzzy_match(text, self.building_material_mapping.keys())
        if match:
            found_key, score = match
            logger.debug(f"Found key: {found_key} with score {score}")
            return self.building_material_mapping[found_key]
        else:
            logger.debug(f"No match found for {text}")
            return None

    def _parse_dedication(self, text: str):
        for key, value in self.dedication_mapping.items():
            if key in text or value in text:
                return value
        logger.debug(f"Fuzzy matching: {text}")
        
        match = self.fuzzy_match(text, self.dedication_mapping.keys())
        if match:
            found_key, score = match
            logger.debug(f"Found key: {found_key} with score {score}")
            return self.dedication_mapping[found_key]
        else:
            logger.debug(f"No match found for {text}")
            return None



    def parse(self, text: str, key: str):
        if key == "building_material":
            return self._parse_building_material(text)
        elif key == "dedication":
            return self._parse_dedication(text)
        else:
            raise ValueError(f"Invalid key: {key}")


    def parse_page(self, page_json: dict):
        """Return a *new* parsed page dictionary, leaving the original untouched.

        A shallow ``dict.copy()`` is not enough because the ``entries`` list (and the
        dictionaries inside it) would still reference the same objects, causing
        in-place mutation of the original *raw* prediction. This resulted in the
        “raw_llm_response” column in the W&B table containing already-parsed
        results. We therefore perform a deep copy so every nested structure is
        duplicated before modification.
        """

        # Import locally to avoid unnecessary dependency at module import time
        from copy import deepcopy

        page_json_copy = deepcopy(page_json)

        for entry in page_json_copy.get("entries", []):
            entry["building_material"] = (
                self.parse(entry.get("building_material", ""), "building_material")
                if entry.get("building_material")
                else None
            )
            entry["dedication"] = (
                self.parse(entry.get("dedication", ""), "dedication")
                if entry.get("dedication")
                else None
            )

        return page_json_copy



