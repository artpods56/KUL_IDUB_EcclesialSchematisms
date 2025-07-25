from dagster import asset
from omegaconf import DictConfig

from core.data.utils import get_dataset
from core.config.helpers import with_configs
from core.config.constants import ConfigType, DatasetConfigSubtype


@asset
@with_configs(
    dataset_config=("schematism_dataset_config", ConfigType.DATASET, DatasetConfigSubtype.EVALUATION)
)
def schematism_dataset(dataset_config: DictConfig):
    return get_dataset(dataset_config)

