import json
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm
from datasets import Value, Sequence, Features, List
from dotenv import load_dotenv
from omegaconf import DictConfig

import core.schemas.configs  # type: ignore
from core.config.constants import ConfigType, DatasetConfigSubtype
from core.config.helpers import with_configs
from core.data.utils import get_dataset
from core.utils.logging import setup_logging
from core.utils.shared import TMP_DIR

setup_logging()

import structlog

logger = structlog.get_logger(__name__)

envs = load_dotenv()
if not envs:
    logger.warning("No environment variables loaded.")


GENERATED_DATASET_DIR = TMP_DIR / "generated" / "dataset"


def _default_source() -> dict[str, object]:
    return {"page_number": None, "entries": []}


def _resolve_source_payload(sample: dict[str, object]) -> dict[str, object]:
    schematism_name = sample.get("schematism_name")
    filename = sample.get("filename")

    if not isinstance(schematism_name, str) or not isinstance(filename, str):
        logger.warning(
            "Missing identifiers for sample; assigning empty source.",
            schematism_name=schematism_name,
            filename=filename,
        )
        return _default_source()

    base_name = Path(f"{schematism_name}_{Path(filename).stem}").stem
    source_path = (GENERATED_DATASET_DIR / base_name).with_suffix(".json")

    if not source_path.exists():
        logger.debug("Source JSON not found; using default.", path=str(source_path))
        return _default_source()

    try:
        with open(source_path, "r", encoding="utf-8") as fh:
            loaded_source = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to decode source JSON; using default.",
            path=str(source_path),
            error=str(exc),
        )
        return _default_source()
    except OSError as exc:
        logger.warning(
            "Failed to read source JSON; using default.",
            path=str(source_path),
            error=str(exc),
        )
        return _default_source()

    if not isinstance(loaded_source, dict):
        logger.warning(
            "Source JSON has unexpected structure; using default.",
            path=str(source_path),
        )
        return _default_source()

    return loaded_source


def _attach_source_column(sample: dict[str, object]) -> dict[str, object]:
    sample["source"] = _resolve_source_payload(sample)
    return sample


source_feature = Features(
    {
        "entries": List(
            {  # 'entries' is a list (Sequence) of dictionaries
                "building_material": Value("string"),
                "deanery": Value("string"),
                "dedication": Value("string"),
                "parish": Value("string"),
            }
        ),
        "page_number": Value("string"),
    }
)


@with_configs(
    dataset_config=(
        "base_schematism_dataset_config",
        ConfigType.DATASET,
        DatasetConfigSubtype.GENERATION,
    )
)
def main(
    dataset_config: DictConfig,
):
    dataset = get_dataset(dataset_config)

    logger.info("Fetching source data.")
    logger.info("Features", features=dataset.features)

    # Using the map function is generally recommended for large datasets
    # as it processes samples one by one.
    # First, define the features for the *entire* resulting dataset
    # new_features = dataset.features.copy()
    # new_features["source"] = source_feature

    dataset = dataset.map(
        _attach_source_column,
        # features=new_features,
        # input_columns=["schematism_name", "filename", "parsed"],
    )

    # payloads = []

    # for sample in dataset.select_columns(["schematism_name", "filename"]):
    #     source_payload = _resolve_source_payload(sample)
    #     payloads.append(source_payload)

    # dataset = dataset.add_column(name="source", column=payloads, feature=source_feature)

    # dataset.column_names = ["image", "source", "parsed", "schematism_name", "filename"]

    logger.info("Finished attaching source data.")

    dataset.push_to_hub(dataset_config.path)


if __name__ == "__main__":
    main()
