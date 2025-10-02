from typing import List, Optional

from pydantic import BaseModel, Field

from core.config.constants import ConfigType, DatasetConfigSubtype
from core.config.registry import register_config


class ColumnMap(BaseModel):
    image_column: str = "image"
    ground_truth_column: str = "ground_truth"
    boxes_column: str = "bboxes"
    labels_column: str = "labels"
    tokens_column: str = "tokens"


@register_config(ConfigType.DATASET, DatasetConfigSubtype.GENERATION)
class BaseDatasetConfig(BaseModel):
    path: str = Field(default="", description="Dataset path or identifier")
    name: str = Field(default="default_dataset", description="Dataset description")
    force_download: bool = Field(default=False, description="Force download the data")
    trust_remote_code: bool = Field(
        default=True, description="Trust remote code when downloading"
    )
    keep_in_memory: bool = Field(default=False, description="Keep data in memory")
    num_proc: int = Field(default=8, description="Number of processes for data loading")
    split: str = Field(default="train", description="Dataset split to use")
    streaming: bool = Field(default=False, description="Enable streaming mode")


class BaseTrainingDatasetConfig(BaseDatasetConfig):
    seed: int = Field(default=42, description="Random seed")
    eval_size: float = Field(default=0.2, description="Evaluation set size")
    test_size: float = Field(default=0.1, description="Test set size")


@register_config(ConfigType.DATASET, DatasetConfigSubtype.TRAINING)
class LayoutLMv3TrainingDatasetConfig(BaseTrainingDatasetConfig):
    column_map: ColumnMap = Field(
        default_factory=ColumnMap,
        description="Map of columns to use for training",
    )


@register_config(ConfigType.DATASET, DatasetConfigSubtype.EVALUATION)
class SchematismsEvaluationDatasetConfig(BaseDatasetConfig):
    column_map: ColumnMap = Field(
        default_factory=ColumnMap,
        description="Mapping of column names to column values",
    )
    full_schematisms: Optional[List[str]] = Field(
        default_factory=list, description="List of schematisms to evaluate"
    )
    partial_schematisms: Optional[List[str]] = Field(
        default_factory=list,
        description="List of schematisms selected partial schematisms",
    )
    positive_samples: int = Field(
        default=10,
        description="List of n first positive samples to fetch from given partial schematism",
    )
    negative_samples: int = Field(
        default=5,
        description="List of n first negative samples to fetch from given partial schematism",
    )
    classes_to_remove: List[str] = Field(
        default_factory=list, description="List of classes to remove from training"
    )
    languages: List[str] = Field(
        default_factory=list, description="List of languages to use"
    )
