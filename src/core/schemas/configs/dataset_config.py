from typing import List, Optional
from pydantic import BaseModel, Field

from core.config.registry import register_config
from core.config.constants import ConfigType, DatasetConfigSubtype

@register_config(ConfigType.DATASET, DatasetConfigSubtype.DEFAULT)
class BaseDatasetConfig(BaseModel):
    path: str = Field(default="", description="Dataset path or identifier")
    name: str = Field(default="default_dataset", description="Dataset name")
    force_download: bool = Field(default=False, description="Force download the dataset")
    trust_remote_code: bool = Field(default=True, description="Trust remote code when downloading")
    keep_in_memory: bool = Field(default=False, description="Keep dataset in memory")
    num_proc: int = Field(default=8, description="Number of processes for data loading")
    split: str = Field(default="train", description="Dataset split to use")
    streaming: bool = Field(default=False, description="Enable streaming mode")

class BaseTrainingDatasetConfig(BaseDatasetConfig):
    seed: int = Field(default=42, description="Random seed")
    eval_size: float = Field(default=0.2, description="Evaluation set size")
    test_size: float = Field(default=0.1, description="Test set size")

@register_config(ConfigType.DATASET, DatasetConfigSubtype.TRAINING)
class LayoutLMv3TrainingDatasetConfig(BaseTrainingDatasetConfig):
    image_column_name: str = Field(default="image_pil", description="Column name for images")
    text_column_name: str = Field(default="words", description="Column name for text")
    boxes_column_name: str = Field(default="bboxes", description="Column name for bounding boxes")
    label_column_name: str = Field(default="labels", description="Column name for labels")

@register_config(ConfigType.DATASET, DatasetConfigSubtype.EVALUATION)
class SchematismsEvaluationDatasetConfig(BaseDatasetConfig):
    image_column_name: str = Field(default="image", description="Column name with encoded image file")
    ground_truth_column_name: str = Field(default="results", description="Column with structured ground truth in JSON format")
    full_schematisms: Optional[List[str]] = Field(default_factory=list, description="List of schematisms to evaluate")
    partial_schematisms: Optional[List[str]] = Field(default_factory=list, description="List of schematisms selected partial schematisms")
    positive_samples: int = Field(default=10, description="List of n first positive samples to fetch from given partial schematism")
    negative_samples: int = Field(default=5, description="List of n first negative samples to fetch from given partial schematism")
    classes_to_remove: List[str] = Field(default_factory=list, description="List of classes to remove from training")
