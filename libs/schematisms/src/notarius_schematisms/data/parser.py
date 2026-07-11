import os
import json
from pathlib import Path
from typing import Any, cast

from notarius_schematisms.domain.models import SchematismPage
from thefuzz import fuzz, process

from structlog import get_logger

from notarius_shared.constants import MAPPINGS_DIR

logger = get_logger(__name__)

DEFAULT_BUILDING_MATERIAL_MAPPING = "building_material.json"
DEFAULT_DEDICATION_MAPPING = "dedication.json"
DEFAULT_DEANERY_MAPPING = "deanery.json"


def _load_mapping(mappings_dir: Path, env_var: str, default_file: str) -> dict[str, str]:
    mapping_path = os.getenv(env_var, default_file)
    with open(mappings_dir / Path(mapping_path), "r") as f:
        return cast(dict[str, str], json.load(f))


class Parser:
    def __init__(
        self,
        mappings_dir: Path | None = None,
        building_material_mapping: dict[str, str] | None = None,
        dedication_mapping: dict[str, str] | None = None,
        deanery_mapping: dict[str, str] | None = None,
        fuzzy_threshold: int = 80,
    ):
        if mappings_dir is None:
            mappings_dir = MAPPINGS_DIR

        if building_material_mapping is None:
            building_material_mapping = _load_mapping(
                mappings_dir,
                "BUILDING_MATERIAL_MAPPINGS",
                DEFAULT_BUILDING_MATERIAL_MAPPING,
            )

        if dedication_mapping is None:
            dedication_mapping = _load_mapping(
                mappings_dir,
                "SAINTS_MAPPINGS",
                DEFAULT_DEDICATION_MAPPING,
            )

        if deanery_mapping is None:
            deanery_mapping = _load_mapping(
                mappings_dir,
                "DEANERY_MAPPINGS",
                DEFAULT_DEANERY_MAPPING,
            )

        self.mappings: dict[str, dict[str, str]] = {
            "dedication": cast(dict[str, str], dedication_mapping),
            "building_material": cast(dict[str, str], building_material_mapping),
            "deanery": cast(dict[str, str], deanery_mapping),
        }

        self.fuzzy_threshold = fuzzy_threshold
        self.fuzzy_scorer = fuzz.ratio

    def fuzzy_match(self, text: str, keys: list[str]) -> tuple[str, int] | None:
        result: Any = process.extractOne(
            text, keys, scorer=self.fuzzy_scorer, score_cutoff=self.fuzzy_threshold
        )
        if not result:
            return None
        # Normalize possible 2- or 3-tuple from thefuzz into (choice, score)
        choice = result[0]
        score = int(result[1])
        return choice, score

    def parse(self, text: str, field_name: str) -> str | None:
        if field_name not in self.mappings:
            raise ValueError(f"Invalid field name: {field_name}")
        else:
            mappings: dict[str, str] = self.mappings[field_name]

        for key, value in mappings.items():
            if key == text:
                return value

        match = self.fuzzy_match(text, list(mappings.keys()))

        if match:
            found_key, score = match
            logger.debug(
                "Fuzzy match used",
                field=field_name,
                input=text,
                match=found_key,
                score=score,
            )
            return mappings[found_key]
        else:
            return None

    def parse_page(self, page_data: SchematismPage) -> SchematismPage:
        """Return a *new* parsed page dictionary, leaving the original untouched.

        A shallow ``dict.copy()`` is not enough because the ``entries`` list (and the
        dictionaries inside it) would still reference the same objects, causing
        in-place mutation of the original *raw* prediction. This resulted in the
        “raw_llm_response” column in the W&B table containing already-parsed
        sample. We therefore perform a deep copy so every nested structure is
        duplicated before modification.
        """

        page_data_dump = page_data.model_dump()

        for entry in page_data_dump["entries"]:
            for field, value in entry.items():
                if field in self.mappings and value:
                    entry[field] = self.parse(value, field)

        return SchematismPage(**page_data_dump)
