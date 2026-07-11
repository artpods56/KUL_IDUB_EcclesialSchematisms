import dagster as dg


class ConfigReference(dg.Config):
    config_name: str
    config_type_name: str
    config_subtype_name: str


class BaseModelConfig(dg.Config):
    enable_cache: bool
