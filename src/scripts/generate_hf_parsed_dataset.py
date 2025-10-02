"""
A script to generate and upload a dataset derived from schematisms to the Hugging Face Hub.

This script utilizes configurations to transform data from schematisms stored in a
specific directory into a dataset and uploads it to a defined repository on the Hugging
Face Hub. The primary tasks involve loading necessary configurations, processing data
entries, and uploading the prepared dataset.

Attributes:
    SCHEMATISMS_DIR (Path): Path to the directory containing specific schematism folders.
    config (ShapefileGeneratorConfig): Configuration object defining paths and key
        column names related to schematism files.
    generator (ShapefileGenerator): Processor responsible for generating data
        from schematism files as per the provided configuration.
    unique_schematisms (list[str]): List of unique schematism names derived from
        folder names within the schematism directory.
    logger (Logger): Structured logger used for capturing script progress and
        other logging information.

Functions:
    dataset_generator():
        Generates dataset samples by iterating through parsed schematisms and
        ensures missing data points are set to None.

Notes:
    Samples saved to the hugging face dataset are structured as follows:
    ```
    {
        "image": PIL.Image object,
        "results": {
            "page_number": int
            "entries": list},
        "schematism_name": str,
        "filename": str,
    }
    ```
"""

import json
from itertools import chain
from pathlib import Path
from typing import cast

import structlog
from datasets import Dataset, Features, List
from datasets.features import Sequence, Value, Image
from dotenv import load_dotenv

from core.data.schematism_parser import ShapefileGenerator, ShapefileGeneratorConfig
from core.utils.shared import REPOSITORY_ROOT, TMP_DIR
from core.utils.logging import setup_logging

setup_logging()
load_dotenv()

logger = structlog.get_logger(__name__)


def main():

    schematism_page_features = Features(
        {
            "entries": List(
                {
                    "building_material": Value("string"),
                    "deanery": Value("string"),
                    "dedication": Value("string"),
                    "parish": Value("string"),
                }
            ),
            "page_number": Value("string"),
        }
    )
    #
    features = Features(
        {
            "image": Image(),
            "source": schematism_page_features,
            "parsed": schematism_page_features,
            "schematism_name": Value("string"),
            "filename": Value("string"),
        }
    )

    SCHEMATISMS_DIR = REPOSITORY_ROOT / Path("data/schematyzmy")
    GENERATED_DATASET_DIR = TMP_DIR / "generated" / "dataset"

    config = ShapefileGeneratorConfig(
        csv_path=REPOSITORY_ROOT / Path("data/csv/dane_hasla_with_filename.csv"),
        schematisms_dir=SCHEMATISMS_DIR,
        relative_shapefile_path=Path("matryca/matryca.shp"),
        schematism_name_column="skany",
        file_name_column="location",
        image_subdir=None,
    )

    generator = ShapefileGenerator(config)

    unique_schematisms = [folder_name.stem for folder_name in SCHEMATISMS_DIR.iterdir()]

    def dataset_generator():
        """
        yield {
            "image": image,
            "results": page_data.model_dump(),
            "schematism_name": schematism_name,
            "filename": image_path.name,
        }
        Returns:

        """

        for sample in chain.from_iterable(
            generator.iter_pages(name) for name in unique_schematisms
        ):

            for entry in sample["parsed"]["entries"]:
                for field, value in entry.items():
                    if value == "[brak_informacji]":
                        entry[field] = None

            yield sample

    logger.info("Generating dataset...")
    dataset = cast(
        Dataset, Dataset.from_generator(dataset_generator, features=features)
    )
    logger.info("Dataset generated", sample_count=len(dataset))

    # logger.info("Saving dataset to Hugging Face Hub...")
    # dataset.push_to_hub(
    #     "artpods56/KUL_IDUB_EcclessiaSchematisms", split="train", max_shard_size="100MB"
    # )


if __name__ == "__main__":
    main()
