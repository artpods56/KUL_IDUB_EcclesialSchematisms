import json
from typing import cast, Tuple, Dict, List, Optional, Callable, Any
from rich.progress import Progress
from abc import ABC, abstractmethod

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


class ProcessingStep(ABC):
    """Abstract base class for processing steps in the evaluation pipeline."""
    
    @abstractmethod
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Process the data and return the modified data."""
        pass


class PreprocessingStep(ProcessingStep):
    """Base class for preprocessing steps."""
    pass


class PostprocessingStep(ProcessingStep):
    """Base class for postprocessing steps."""
    pass


class EvaluationStep(ProcessingStep):
    """Base class for evaluation steps."""
    pass


class SampleFilterStep(PreprocessingStep):
    """Filter samples based on positive/negative sample counts."""
    
    def __init__(self, positive_samples: Optional[int], negative_samples: Optional[int]):
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples
    
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        results_json = data.get("results_json", {})
        
        if len(results_json.get("entries", [])) == 0:
            if self.negative_samples == 0:
                return None  # Skip this sample
            self.negative_samples -= 1
        else:
            if self.positive_samples == 0:
                return None  # Skip this sample
            self.positive_samples -= 1
        
        return data


class PredictionStep(ProcessingStep):
    """Generate predictions using the models."""
    
    def __init__(self, lmv3_model: LMv3Model, llm_model: LLMModel, ocr_model, parser: Parser):
        self.lmv3_model = lmv3_model
        self.llm_model = llm_model
        self.ocr_model = ocr_model
        self.parser = parser
    
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        image = data["image"]
        metadata = data["metadata"]
        
        # Generate predictions
        lmv3_prediction = self.lmv3_model.predict(image, **metadata)
        ocr_text = self.ocr_model.predict(image, text_only=True, **metadata)
        llm_prediction = self.llm_model.predict(image=image, hints=lmv3_prediction, text=ocr_text, **metadata)
        parsed_llm_prediction = self.parser.parse_page(llm_prediction)
        
        # Add predictions to data
        data.update({
            "lmv3_prediction": lmv3_prediction,
            "ocr_text": ocr_text,
            "llm_prediction": llm_prediction,
            "parsed_llm_prediction": parsed_llm_prediction
        })
        
        return data


class MetricsCalculationStep(EvaluationStep):
    """Calculate evaluation metrics."""
    
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        results_json = data["results_json"]
        parsed_llm_prediction = data["parsed_llm_prediction"]
        
        metrics = evaluate_json_response(results_json, parsed_llm_prediction)
        data["metrics"] = metrics
        
        return data


class WandbLoggingStep(EvaluationStep):
    """Log metrics to wandb."""
    
    def __init__(self, wandb_run, default_fields: Tuple[str, ...]):
        self.wandb_run = wandb_run
        self.default_fields = default_fields
        self._global_step = 0
    
    def process(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        metrics = data["metrics"]
        
        # Log scalars to wandb
        scalar_log = {}
        for fld in self.default_fields:
            fld_metrics = metrics.get(fld, {})
            for m in ("precision", "recall", "f1", "accuracy"):
                scalar_log[f"{fld}/{m}"] = float(fld_metrics.get(m, 0.0))
        
        self.wandb_run.log(scalar_log, step=self._global_step)
        self._global_step += 1
        
        return data


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
        
        # Initialize processing steps
        self.preprocessing_steps: List[PreprocessingStep] = []
        self.postprocessing_steps: List[PostprocessingStep] = []
        self.evaluation_steps: List[EvaluationStep] = []
        
        # Add default steps
        self._setup_default_steps()

    def _setup_default_steps(self):
        """Setup the default processing steps."""
        # Preprocessing steps
        self.add_preprocessing_step(SampleFilterStep(self.positive_samples, self.negative_samples))
        
        # Postprocessing steps
        self.add_postprocessing_step(PredictionStep(self.lmv3_model, self.llm_model, self.ocr_model, self.parser))
        
        # Evaluation steps
        self.add_evaluation_step(MetricsCalculationStep())

    def add_preprocessing_step(self, step: PreprocessingStep):
        """Add a preprocessing step to the pipeline."""
        self.preprocessing_steps.append(step)

    def add_postprocessing_step(self, step: PostprocessingStep):
        """Add a postprocessing step to the pipeline."""
        self.postprocessing_steps.append(step)

    def add_evaluation_step(self, step: EvaluationStep):
        """Add an evaluation step to the pipeline."""
        self.evaluation_steps.append(step)

    def preprocess_sample(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply preprocessing steps to a sample."""
        data = {
            "filename": sample["filename"],
            "schematism_name": sample["schematism_name"],
            "image": sample[self.dataset_config.image_column_name],
            "results": sample[self.dataset_config.ground_truth_column_name],
            "metadata": {
                "schematism": sample["schematism_name"],
                "filename": sample["filename"],
            }
        }
        
        # Parse ground truth
        results = data["results"].replace('"[brak_informacji]"', "null")
        data["results_json"] = json.loads(results)
        
        # Apply preprocessing steps
        for step in self.preprocessing_steps:
            data = step.process(data)
            if data is None:
                return None  # Sample was filtered out
        
        return data

    def postprocess_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply postprocessing steps to a sample."""
        for step in self.postprocessing_steps:
            data = step.process(data)
        return data

    def evaluate_sample(self, data: Dict[str, Any], wandb_run) -> Dict[str, Any]:
        """Apply evaluation steps to a sample."""
        for step in self.evaluation_steps:
            if isinstance(step, WandbLoggingStep):
                step.wandb_run = wandb_run
            data = step.process(data)
        return data

    def calculate_metrics(self, preds_and_gt_pairs):
        for pred, gt in preds_and_gt_pairs:
            metrics = evaluate_json_response(gt, pred)
            logger.info("Evaluation metrics:", metrics=metrics)

    def run(self, dataset_subset: Dataset, logger, schematism_name: str, wandb_run):
        eval_table = create_eval_table()
        
        # Add wandb logging step
        wandb_logging_step = WandbLoggingStep(wandb_run, _DEFAULT_FIELDS)
        self.add_evaluation_step(wandb_logging_step)

        with Progress() as progress:
            task = progress.add_task("Evaluating", total=len(dataset_subset))
            progress.start_task(task)
            
            while self.positive_samples > 0 or self.negative_samples > 0 or self.positive_samples is None or self.negative_samples is None:
                for sample in dataset_subset:
                    # Preprocessing
                    data = self.preprocess_sample(sample)
                    if data is None:
                        continue  # Sample was filtered out
                    
                    # Postprocessing (prediction)
                    data = self.postprocess_sample(data)
                    
                    # Evaluation
                    data = self.evaluate_sample(data, wandb_run)
                    
                    # Add to evaluation table
                    add_eval_row(
                        table=eval_table,
                        sample_id=data["filename"],
                        pil_image=data["image"],
                        page_info_json=data["results_json"],
                        lmv3_response=cast(Dict, data["lmv3_prediction"]),
                        raw_llm_response=data["llm_prediction"],
                        parsed_llm_response=data["parsed_llm_prediction"],
                        metrics=data["metrics"]
                    )
                    
                    # Log results
                    logger.info("LLM Predictions:", filename=data["filename"], llm_prediction=data["llm_prediction"])
                    logger.info("Ground truth:", filename=data["filename"], results_json=data["results_json"])
                    logger.info("Evaluation metrics:", filename=data["filename"], metrics=data["metrics"])
                    
                    progress.update(task, advance=1)

        summary_table = create_summary_table(eval_table)

        wandb_run.log({
            f"{schematism_name}_eval_table": eval_table,
            f"{schematism_name}_summary": summary_table,
        })
