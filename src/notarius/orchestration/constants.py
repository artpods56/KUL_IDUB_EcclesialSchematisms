from typing import final


@final
class AssetLayer:
    STG = "stg"  # Staging - raw ingestion (Job 1: ingest)
    INT = "int"  # Intermediate - transformations (Job 2: transform)
    FCT = "fct"  # Facts - predictions/events (Job 3: predict)
    DIM = "dim"  # Dimensions - reference data
    MRT = "mrt"  # Marts - exports/final outputs (Job 4: export)
    RES = "res"  # Resources - ml_models/utilities/resources


@final
class Kinds:
    PYTHON = "python"
    HUGGINGFACE = "huggingface"
    PYDANTIC = "pydantic"
    PANDAS = "pandas"
    YAML = "yaml"
    JSON = "json"
    EXCEL = "excel"
    WANDB = "wandb"


@final
class DataSource:
    HUGGINGFACE = Kinds.HUGGINGFACE
    FILE = "file"
    DATABASE = "database"
    MIXED = "mixed"
    PREDICTION = "prediction"


@final
class ResourceGroup:
    MODEL = "model"
    CONFIG = "config_manager"
    DATA = "data"
    EVALUATION = "evaluation"
