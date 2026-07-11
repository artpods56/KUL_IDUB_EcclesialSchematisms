from typing import Any, Mapping

import dagster as dg
from dagster import AssetIn, AssetExecutionContext, MetadataValue
from pydantic import BaseModel

from notarius.application.ports.outbound.engine import ConfigurableEngine
from notarius.infrastructure.config.manager import config_manager
from notarius.infrastructure.config.constants import ConfigType, ModelsConfigSubtype
from notarius.infrastructure.llm.engine_adapter import LLMEngine
from notarius.infrastructure.ml_models.lmv3.engine_adapter import LMv3Engine
from notarius.infrastructure.ocr.engine_adapter import OCREngine
from notarius_dagster.constants import AssetLayer, ResourceGroup, Kinds


def asset_factory__model[ModelT: ConfigurableEngine[Any], ConfigT: BaseModel](
    asset_name: str,
    ins: Mapping[str, AssetIn],
    model_class: type[ModelT],
    model_config: ConfigT,
    extra_kwargs: dict[str, Any] | None = None,
    key_prefix: list[str] | None = None,
):
    @dg.asset(
        name=asset_name,
        key_prefix=key_prefix or [AssetLayer.RES],
        group_name=ResourceGroup.MODEL,
        kinds={Kinds.PYTHON},
        ins=ins,  # {"model_config": AssetIn(key=[AssetLayer.RES, model_config])},
        # io_manager_key="mem_io_manager",
    )
    def _asset__model(context: AssetExecutionContext):
        return model_class.from_config(model_config)

    return _asset__model


ocr_model = asset_factory__model(
    asset_name="ocr_model",
    ins={},
    model_class=OCREngine,
    model_config=config_manager.load_config_as_model(
        config_name="ocr_model_config",
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.OCR,
    ),
)

lmv3_model = asset_factory__model(
    asset_name="lmv3_model",
    ins={},
    model_class=LMv3Engine,
    model_config=config_manager.load_config_as_model(
        config_name="lmv3_model_config",
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.LMV3,
    ),
)

llm_model = asset_factory__model(
    asset_name="llm_model",
    ins={},
    model_class=LLMEngine,
    model_config=config_manager.load_config_as_model(
        config_name="llm_model_config",
        config_type=ConfigType.MODELS,
        config_subtype=ModelsConfigSubtype.LLM,
    ),
)


@dg.asset(
    key_prefix=[AssetLayer.RES],
    group_name=ResourceGroup.MODEL,
    kinds={Kinds.PYTHON},
)
def parser(context: AssetExecutionContext):
    """Create a parser instance for translating/normalizing schematism entries.

    This asset creates a Parser instance that can perform fuzzy matching
    and translation of dedications, building materials, and deaneries.
    """
    from notarius.domain.services.parser import Parser

    parser_instance = Parser()

    context.add_asset_metadata(
        {
            "fuzzy_threshold": MetadataValue.int(parser_instance.fuzzy_threshold),
            "mapping_keys": MetadataValue.json(list(parser_instance.mappings.keys())),
        }
    )

    return parser_instance
