import dagster as dg

from dagster_graphql import DagsterGraphQLClient

from notarius.config import DagsterConfig
from notarius.infrastructure.llm.utils import parse_model_name
from notarius.orchestration.assets.load.export import (
    eval__excel_export_parsed_dataframe__pandas,
    eval__excel_export_source_dataframe__pandas,
    pred__export_llm_enriched_dataset__json,
    ParsedDataFrameExportConfig,
    SourceDataFrameExportConfig,
    PredsSourceExportConfig,
)
from notarius.orchestration.assets.transform.predict import (
    pred__llm_enriched_dataset__pydantic,
    LLMConfig,
)
from notarius.orchestration.assets.transform.preprocess import (
    PreprocessingConfig,
    preprocessed__hf__dataset,
)

from notarius.orchestration.pipelines.evaluation import (
    ALL_EVALUATION_ASSETS_WITH_CONFIGS,
)

from notarius.orchestration.constants import JobType, Environment

from notarius.shared.logger import get_logger

logger = get_logger(__name__)


MODELS = [
    # "qwen/qwen3-vl-30b-a3b-instruct", # run and will finish, will not, lol
    # "qwen/qwen3-vl-8b-thinking", # not working, probably due to no structured output support
    # "baidu/ernie-4.5-vl-28b-a3b", # does not support structured output
    # "mistralai/mistral-medium-3.1",
    # "amazon/nova-pro-v1", # failed to produce structured output
    # "nvidia/nemotron-nano-12b-v2-vl:free",
    # "thudm/glm-4.1v-9b-thinking",
    # "stepfun-ai/step3", # throws exception for some reason
    # vVv working models vVv
    "google/gemini-3-flash-preview",  # our best friend
    # "google/gemma-3-12b-it",  # going strong
    # "bytedance-seed/seed-1.6-flash"  # testing
    # "z-ai/glm-4.6v" # fails to generate structured output, would need json healing
    "qwen/qwen3-vl-32b-instruct",  # testing
]


dagster_config = DagsterConfig()


client = DagsterGraphQLClient(
    hostname=dagster_config.host, port_number=dagster_config.port
)


def main():

    for model in MODELS:

        default_config = ALL_EVALUATION_ASSETS_WITH_CONFIGS.copy()

        default_config.update(
            {
                preprocessed__hf__dataset: {
                    "config": PreprocessingConfig(
                        filtered_schematisms=[
                            "wloclawek_1873",
                            "tarnow_1870",
                        ]
                    ).model_dump()
                }
            }
        )

        default_config.update(
            {
                pred__llm_enriched_dataset__pydantic: {
                    "config": LLMConfig(model_name=model).model_dump()
                }
            }
        )

        default_config.update(
            {
                eval__excel_export_parsed_dataframe__pandas: {
                    "config": ParsedDataFrameExportConfig(
                        file_name=f"{parse_model_name(model)}_parsed_schematism_comp.xlsx"
                    ).model_dump()
                },
                eval__excel_export_source_dataframe__pandas: {
                    "config": SourceDataFrameExportConfig(
                        file_name=f"{parse_model_name(model)}_source_schematism_comp.xlsx"
                    ).model_dump()
                },
                pred__export_llm_enriched_dataset__json: {
                    "config": PredsSourceExportConfig(
                        filename_prefix=f"predictions_{parse_model_name(model)}"
                    ).model_dump()
                },
            }
        )

        run_id = client.submit_job_execution(
            job_name="evaluation_pipeline",
            run_config=dg.RunConfig(
                ops={
                    asset.key.to_python_identifier(): config
                    for asset, config in default_config.items()
                    if config is not None
                },
            ),
            tags={
                "environment": Environment.DEV,
                "task": JobType.EVALUATION,
                "model": model,
            },
        )

        logger.info(f"Submitted run for {model}: {run_id}")


if __name__ == "__main__":
    main()
