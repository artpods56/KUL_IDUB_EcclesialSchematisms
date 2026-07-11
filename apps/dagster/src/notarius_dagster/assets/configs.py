from typing import Sequence, cast, Any
import dagster as dg
from dagster import AssetExecutionContext, MetadataValue
from omegaconf import OmegaConf
from pydantic import BaseModel

from notarius.infrastructure.config.manager import ConfigManager
from notarius_dagster.configs.shared import ConfigReference
from notarius_dagster.constants import AssetLayer, Kinds, ResourceGroup
from notarius.schemas.configs import (
    PytesseractOCRConfig,
    BaseLMv3ModelConfig,
    LLMEngineConfig,
)
from notarius.schemas.configs.dataset_config import BaseDatasetConfig


def asset_factory__config(
    asset_name: str,
    config_model: type[BaseModel],
    key_prefix: Sequence[str] | None = None,
):
    """Factory function to create config_manager loading assets."""

    @dg.asset(
        name=asset_name,
        key_prefix=key_prefix or [AssetLayer.RES],
        group_name=ResourceGroup.CONFIG,
        kinds={Kinds.PYTHON, Kinds.YAML},
    )
    def _asset__config(
        context: AssetExecutionContext,
        config: ConfigReference,
        config_manager: dg.ResourceParam[ConfigManager],
    ):
        config_reference = {
            "config_name": config.config_name,
            "config_type_name": config.config_type_name,
            "config_subtype_name": config.config_subtype_name,
        }

        omega_config = config_manager.load_config_from_string(**config_reference)

        config_dict = cast(
            dict[str, Any], OmegaConf.to_container(omega_config, resolve=True)
        )
        context.add_asset_metadata(
            {"config_reference": MetadataValue.json(config_reference)}
        )

        context.add_output_metadata({"config_manager": MetadataValue.json(config_dict)})

        return config_model(**config_dict)

    return _asset__config


# --- dataset configs ---
hf_dataset__config = asset_factory__config("hf_dataset__config", BaseDatasetConfig)

# --- model configs ---
ocr_model__config = asset_factory__config("ocr_model__config", PytesseractOCRConfig)
lmv3_model__config = asset_factory__config("lmv3_model__config", BaseLMv3ModelConfig)
llm_model__config = asset_factory__config("llm_model__config", LLMEngineConfig)
