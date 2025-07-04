# src/my_project/utils/config.py
from collections.abc import Mapping as ABCMapping
from typing import Any, Dict

from omegaconf import OmegaConf


def config_to_dict(cfg) -> Dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, ABCMapping):
        raise TypeError("Expected OmegaConf root to be a mapping")
    return dict(container)
