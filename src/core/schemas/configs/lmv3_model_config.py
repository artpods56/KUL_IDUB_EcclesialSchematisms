from pydantic import BaseModel, Field

from core.config.registry import register_config
from core.config.constants import ConfigType, ModelsConfigSubtype

class ModelConfig(BaseModel):
    checkpoint: str = Field(default="microsoft/layoutlmv3-base", description="Model checkpoint path or identifier")

class RunConfig(BaseModel):
    device: str = Field(default="cpu", description="Device to run the model on (cpu, cuda, etc.)")

class ProcessorConfig(BaseModel):
    checkpoint: str = Field(default="microsoft/layoutlmv3-base", description="Processor checkpoint path or identifier")
    max_length: int = Field(default=512, description="Maximum sequence length")
    local_files_only: bool = Field(default=False, description="Enable local files only")


class FocalLossConfig(BaseModel):
    alpha: float = Field(default=1.0, description="Alpha parameter for focal loss")
    gamma: float = Field(default=1.0, description="Gamma parameter for focal loss")


class TrainingConfig(BaseModel):
    output_dir: str = Field(default="./output", description="Output directory for model checkpoints")
    max_steps: int = Field(default=100, description="Maximum number of training steps")
    per_device_train_batch_size: int = Field(default=1, description="Batch size per device for training")
    per_device_eval_batch_size: int = Field(default=1, description="Batch size per device for pipeline")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    eval_strategy: str = Field(default="steps", description="Evaluation strategy")
    eval_steps: int = Field(default=100, description="Evaluation steps")
    load_best_model_at_end: bool = Field(default=True, description="Load best model at end of training")
    metric_for_best_model: str = Field(default="eval_overall_f1", description="Metric to use for best model selection")
    report_to: str = Field(default="wandb", description="Where to report training results")
    run_name: str = Field(default="layoutlmv3-large-focal", description="Run description")
    logging_strategy: str = Field(default="steps", description="Logging strategy")
    logging_steps: int = Field(default=100, description="Logging steps")
    logging_dir: str = Field(default="logs", description="Logging directory")
    save_strategy: str = Field(default="steps", description="Save strategy")
    save_steps: int = Field(default=100, description="Save steps")
    save_total_limit: int = Field(default=4, description="Maximum number of checkpoints to keep")
    fp16: bool = Field(default=True, description="Use mixed precision training")


class InferenceConfig(BaseModel):
    checkpoint: str = Field(default="microsoft/layoutlmv3-base", description="Checkpoint path or identifier for inference")
    apply_ocr: bool = Field(default=False, description="Apply OCR during inference")
    local_files_only: bool = Field(default=False, description="Enable local files only")


class MetricsConfig(BaseModel):
    return_entity_level_metrics: bool = Field(default=True, description="Return entity level metrics")


@register_config(ConfigType.MODELS, ModelsConfigSubtype.LMV3)
class BaseLMv3ModelConfig(BaseModel):

    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    run: RunConfig = Field(default_factory=RunConfig, description="Run configuration")
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig, description="Processor configuration")
    focal_loss: FocalLossConfig = Field(default_factory=FocalLossConfig, description="Focal loss configuration")
    training: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    inference: InferenceConfig = Field(default_factory=InferenceConfig, description="Inference configuration")
    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Metrics configuration")
