from typing import Dict, List

import structlog
from wandb import Run

from core.pipeline.steps.base import DatasetProcessingStep
from core.schemas.data.pipeline import PipelineData
from core.utils.wandb_eval import (
    add_eval_row,
    create_eval_table,
    create_summary_table,
)

logger = structlog.get_logger(__name__)


class WandbLoggingStep(DatasetProcessingStep[list[PipelineData], list[PipelineData]]):
    """
    Log evaluation results to Weights & Biases tables with optional grouping.

    Features:
    - Logs a per-sample evaluation table (tp/fp/fn/support/precision/recall/f1/accuracy)
      for standard fields returned by evaluation.
    - Optionally groups samples by a chosen metadata key creating one table per group.
    - Logs a summary table per group with mean/min/max for selected metrics.
    - Additionally logs an overall-mean-per-metric table (averaged across fields).

    Args:
        wandb_run: Active W&B run to log to.
        group_by_metadata_key: Metadata key to split samples into separate tables. If ``None``,
            all samples are logged into a single table named ``"eval/all"``.
        fields: Optional override for metric fields (defaults taken by helpers).
        table_prefix: Artifact key prefix for logged tables.
        log_summary: Whether to create and log summary tables alongside per-sample tables.
    """

    def __init__(
        self,
        wandb_run: Run,
        group_by_metadata_key: str | None = None,
        fields: List[str] | None = None,
        table_prefix: str = "eval",
        log_summary: bool = True,
    ):
        super().__init__()
        self.wandb_run = wandb_run
        self.group_by_metadata_key = group_by_metadata_key
        self.fields = fields
        self.table_prefix = table_prefix.rstrip("/")
        self.log_summary = log_summary


    def _group_dataset(self, dataset: list[PipelineData]) -> Dict[str, list[PipelineData]]:
        """Groups dataset by metadata key."""
        if not self.group_by_metadata_key:
            return {"all": dataset}

        groups: Dict[str, list[PipelineData]] = {}
        for item in dataset:

            group_key = item.metadata.get(self.group_by_metadata_key, None)
            if group_key is None:
                raise ValueError(f"Invalid metadata key, available keys: {item.metadata.keys()}")

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(item)
        return groups

    def process_dataset(self, dataset: list[PipelineData]) -> list[PipelineData]:
        groups = self._group_dataset(dataset)

        for group_name, items in groups.items():
            table_name = f"{self.table_prefix}/{group_name}"

            eval_table = create_eval_table(fields=self.fields)

            for item in items:
                add_eval_row(
                    table=eval_table,
                    pipeline_data=item,
                    fields=self.fields
                )

            if self.log_summary and len(eval_table.data) > 0:
                try:
                    summary_table = create_summary_table(eval_table)
                    self.wandb_run.log({f"{table_name}_summary": summary_table})
                except Exception as e:
                    logger.error("Failed to create summary table", error=e)
            
            self.wandb_run.log({table_name: eval_table})
      
        return dataset
