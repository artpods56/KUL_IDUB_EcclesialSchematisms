from typing import List
from pydantic import BaseModel, Field, ConfigDict

from core.config.registry import register_config
from core.config.constants import ConfigType, WandbConfigSubtype

@register_config(ConfigType.WANDB, WandbConfigSubtype.DEFAULT)
class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enable: bool = Field(default=True, description="Enable Weights & Biases logging")
    project: str = Field(default="ai-osrodek", description="Weights & Biases project description")
    entity: str = Field(default="", description="Weights & Biases entity description")
    name: str = Field(default="experiment", description="Weights & Biases run description")
    tags: List[str] = Field(default_factory=list, description="Tags for the run")
    log_predictions: bool = Field(default=True, description="Log predictions_data")
    num_prediction_samples: int = Field(default=15, description="Number of prediction samples to log")

