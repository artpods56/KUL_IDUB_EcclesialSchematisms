"""
Evaluation steps for the pipeline.

This module provides concrete implementations of evaluation steps for the pipeline.
It includes a detailed evaluation step that calculates precision, recall, F1, and accuracy for each field in the predictions_data.
"""

from core.pipeline.steps.base import SampleProcessingStep
from core.schemas.data.pipeline import PipelineData
from core.data.metrics import evaluate_json_response


class SampleEvaluationStep(SampleProcessingStep[PipelineData, PipelineData]):
    """
    An evaluation step that calculates detailed metrics between ground truth and predictions_data.
    """
    def __init__(self, predictions_source: str = "llm_prediction", ground_truth_source: str = "source_ground_truth"):
        super().__init__()
        self.predictions_source = predictions_source
        self.ground_truth_source = ground_truth_source

    def process_sample(self, data: PipelineData) -> PipelineData:
        """
        Calculates precision, recall, F1, and accuracy for each field in the predictions_data.
        """
        if data.ground_truth is None or data.parsed_prediction is None:
            self.logger.warning("Missing ground_truth or llm_prediction, skipping evaluation.")
            return data

        ground_truth_data = getattr(data, self.ground_truth_source, None)
        predictions_data = getattr(data, self.predictions_source, None)

        if ground_truth_data is None or predictions_data is None:
            self.logger.warning(f"Missing data in sources: {self.ground_truth_source} or {self.predictions_source}, skipping evaluation.")
            return data

        data.evaluation_results = evaluate_json_response(
            predictions_data=predictions_data,
            ground_truth_data=ground_truth_data
        )

        return data