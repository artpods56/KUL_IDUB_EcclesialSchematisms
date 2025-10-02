from typing import List, Optional, Type

import wandb
from pydantic import BaseModel
from wandb import Table

from core.schemas.data.pipeline import PipelineData

DEFAULT_FIELDS = ["page_number", "parish", "deanery", "dedication", "building_material"]

def create_table_from_pydantic(model_class: Type[BaseModel], fields: list[str] | None = None) -> Table:
    fields = fields or DEFAULT_FIELDS
    pydantic_fields = list(model_class.model_fields.keys())
    columns = pydantic_fields

    for field in fields:
        columns.extend([f"{field}_precision", f"{field}_recall", f"{field}_f1"])
    return wandb.Table(columns=columns)

def create_eval_table(fields: Optional[List[str]] = None) -> wandb.Table:
    """Creates a wandb.Table for evaluation results."""
    fields = fields or DEFAULT_FIELDS
    columns = ["image", "parsed_messages", "llm_prediction", "source", "parsed", "parsed_prediction", "lmv3_prediction",]
    for field in fields:
        columns.extend([f"{field}_precision", f"{field}_recall", f"{field}_f1"])
    return wandb.Table(columns=columns)

def add_eval_row(table: wandb.Table, pipeline_data: PipelineData, fields: Optional[List[str]] = None):
    """Adds a row to the evaluation table from PipelineData."""
    fields = fields or DEFAULT_FIELDS
    
    image = wandb.Image(pipeline_data.image.resize((400, 600))) if pipeline_data.image else None
    
    # Build row with simple fields to avoid nested dict type issues in wandb
    row = [
        image,
        pipeline_data.parsed_messages,
        pipeline_data.llm_prediction.model_dump_json() if pipeline_data.llm_prediction else None,
        pipeline_data.source_ground_truth.model_dump_json() if pipeline_data.source_ground_truth else None,
        pipeline_data.ground_truth.model_dump_json() if pipeline_data.ground_truth else None,
        pipeline_data.parsed_prediction.model_dump_json() if pipeline_data.parsed_prediction else None,
        pipeline_data.lmv3_prediction.model_dump_json() if pipeline_data.lmv3_prediction else None,
    ]
    
    # Add metrics
    for field in fields:
        metrics = getattr(pipeline_data.evaluation_results, field, None)
        if metrics:
            row.extend([metrics.precision, metrics.recall, metrics.f1])
        else:
            row.extend([None, None, None])
            
    table.add_data(*row)

def create_summary_table(eval_table: wandb.Table) -> wandb.Table:
    """Creates a summary table with stats for each metric."""
    df = eval_table.get_dataframe()
    summary_data = []
    
    metric_cols = [col for col in df.columns if any(metric in col for metric in ["_precision", "_recall", "_f1"])]
    
    for col in metric_cols:
        mean = df[col].mean()
        std = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        summary_data.append([col, mean, std, min_val, max_val])
        
    summary_table = wandb.Table(data=summary_data, columns=["metric", "mean", "std", "min", "max"])
    return summary_table
