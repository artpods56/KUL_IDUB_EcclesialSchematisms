from typing import Callable

import dagster as dg
from datasets import Dataset
from pydantic import Field

from notarius_dagster.constants import (
    AssetLayer,
    DataSource,
    ResourceGroup,
    Kinds,
)
from notarius.schemas.data.dataset import SchematismPage


def filter_by_schematism(to_filter: str | list[str]) -> Callable[[str], bool]:
    targets = {to_filter} if isinstance(to_filter, str) else set(to_filter)
    return lambda name: name in targets


def filter_empty_entries(sample: SchematismPage) -> bool:
    return bool(sample.get("entries", []))


class PreprocessingConfig(dg.Config):
    """Configuration for filtering by schematism names."""

    filtered_schematisms: list[str] = Field(
        default=[], description="Filter by schematism names."
    )

    filtered_empty_pages: str | None = Field(
        default=None,
        description="Filter samples without entries from either source or parsed.",
    )
    filtered_empty_pages_field: str = Field(
        default="parsed",
        description="Column name with schematisms to use for filtering. Choices: 'source', 'parsed'.",
    )


@dg.asset(
    key_prefix=[AssetLayer.INT, DataSource.HUGGINGFACE],
    group_name=ResourceGroup.DATA,
    kinds={Kinds.PYTHON, Kinds.HUGGINGFACE},
    ins={"dataset": dg.AssetIn(key="raw__hf__dataset")},
    io_manager_key="hf_dataset_io_manager",
)
def preprocessed__hf__dataset(
    context: dg.AssetExecutionContext,
    dataset: Dataset,
    config: PreprocessingConfig,
) -> Dataset:

    if config.filtered_schematisms:
        dataset = dataset.filter(
            filter_by_schematism(config.filtered_schematisms),
            input_columns=["schematism_name"],
        )

    if config.filtered_empty_pages:
        dataset = dataset.filter(
            filter_empty_entries, input_columns=config.filtered_empty_pages_field
        )

    return dataset
