import json
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from omegaconf import DictConfig
from structlog import get_logger

from core.schemas.pipeline.data import PipelineData

logger = get_logger(__name__)


class BaseIngestionAdapter(ABC):
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> PipelineData:
        pass

    @abstractmethod
    def batch_process(self, data: List[Dict[str, Any]]) -> List[PipelineData]:
        pass


class HFIngestionAdapter(BaseIngestionAdapter):

    def __init__(self, dataset_config: DictConfig):
        column_map = dataset_config.get("column_map", {})

        self.image_src_col = column_map.get("image_column")
        self.ground_truth_src_col = column_map.get("ground_truth_column")

        if not self.image_src_col or not self.ground_truth_src_col:
            raise ValueError("Config's column_map must specify 'image_column' and 'ground_truth_column'.")

    def process(self, data: Dict[str, Any], **kwargs) -> PipelineData:
        image = data.get(self.image_src_col)
        ground_truth_str = data.get(self.ground_truth_src_col)

        if image is None or ground_truth_str is None:
            raise ValueError(
                f"Missing required data. Looking for '{self.image_src_col}' and '{self.ground_truth_src_col}' in the data sample.")

        try:
            ground_truth = json.loads(ground_truth_str)
        except json.JSONDecodeError:
            logger.error("Failed to parse ground_truth JSON string.", value=ground_truth_str)
            raise ValueError(f"Invalid JSON in ground_truth for column '{self.ground_truth_src_col}'.")

        metadata = {
            "schematism": data.get("schematism_name"),
            "filename": data.get("filename"),
        }

        mapped_data = {
            "image": image,
            "ground_truth": ground_truth,
            "metadata": metadata,
        }

        return PipelineData(**mapped_data)

    def batch_process(self, data: List[Dict[str, Any]], **kwargs) -> List[PipelineData]:
        return [self.process(item, **kwargs) for item in data]
