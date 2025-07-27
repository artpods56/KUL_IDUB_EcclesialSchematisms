import json
from typing import cast, Tuple, Dict
from rich.progress import Progress



from datasets import Dataset
from omegaconf import DictConfig

from core.models.llm.model import LLMModel
from core.models.lmv3.model import LMv3Model

from core.data.schematism_parser import SchematismParser
from core.data.translation_parser import Parser
from core.models.lmv3.model import ocr_page
from core.data.stats import evaluate_json_response

# wandb integration
import wandb
from core.utils.wandb_eval import create_eval_table, add_eval_row, create_summary_table

_DEFAULT_FIELDS: Tuple[str, ...] = (
    "page_number",
    "deanery",
    "parish",
    "dedication",
    "building_material",
)

from tqdm.contrib.logging import logging_redirect_tqdm
from structlog import get_logger
logger = get_logger(__name__)

class EvaluationRunner:
    def __init__(self, llm_model: LLMModel, lmv3_model: LMv3Model, ocr_model, dataset_config: DictConfig, parser: Parser):
        self.llm_model = llm_model
        self.lmv3_model = lmv3_model
        self.ocr_model = ocr_model
        self.dataset_config = dataset_config
        self.parser = parser
        self.logger = logger.bind(dataset_name=dataset_config.name)
        self._global_step = 0

        self.positive_samples = self.dataset_config.get("positive_samples", None)
        self.negative_samples = self.dataset_config.get("negative_samples", None)


    def calculate_metrics(self, preds_and_gt_pairs):
        for pred, gt in preds_and_gt_pairs:
            metrics = evaluate_json_response(gt, pred)
            logger.info("Evaluation metrics:", metrics=metrics)

    def run(self, dataset_subset: Dataset, logger, schematism_name: str, wandb_run):

        eval_table = create_eval_table()

        for sample in dataset_subset:
            filename = sample["filename"]  # type: ignore[index]
            schematism_name = sample["schematism_name"]  # type: ignore

            metadata = {
                "schematism": schematism_name,
                "filename": filename,
            }

            image = sample[self.dataset_config.image_column_name]  # type: ignore[index]
            results = sample[self.dataset_config.ground_truth_column_name]  # type: ignore[index]
            results = results.replace('"[brak_informacji]"', "null")
            results_json = json.loads(results)

            if len(results_json["entries"]) == 0:
                continue
            #     if self.negative_samples == 0:
            #         continue
            #     self.negative_samples -= 1
            # else:
            #     if self.positive_samples == 0:
            #         continue
            #     self.positive_samples -= 1

            lmv3_prediction = self.lmv3_model.predict(image, **metadata)

            ocr_text = self.ocr_model.predict(image, text_only=True, **metadata)
            llm_prediction = self.llm_model.predict(image=image, hints=lmv3_prediction, text=ocr_text, **metadata)
            parsed_llm_prediction = self.parser.parse_page(llm_prediction)

            metrics = evaluate_json_response(results_json, parsed_llm_prediction)

            # scalars for wandb
            scalar_log = {}
            for fld in _DEFAULT_FIELDS:
                fld_metrics = metrics.get(fld, {})
                for m in ("precision", "recall", "f1", "accuracy"):
                    scalar_log[f"{fld}/{m}"] = float(fld_metrics.get(m, 0.0))

            wandb_run.log(scalar_log, step=self._global_step)



            add_eval_row(
                table = eval_table,
                sample_id = filename,
                pil_image = image,
                page_info_json = results_json,
                lmv3_response = cast(Dict, lmv3_prediction),
                raw_llm_response = llm_prediction,
                parsed_llm_response = parsed_llm_prediction,
                metrics = metrics
            )

            # logger.info("LLM Predictions:", filename=filename, llm_prediction=llm_prediction)
            # logger.info("Ground truth:", filename=filename, results_json=results_json)
            # logger.info(f"Evaluation metrics:", filename=filename, metrics=metrics)

            self._global_step += 1

        summary_table = create_summary_table(eval_table)

        wandb_run.log({
            f"{schematism_name}_eval_table": eval_table,
            f"{schematism_name}_summary": summary_table,
        })
