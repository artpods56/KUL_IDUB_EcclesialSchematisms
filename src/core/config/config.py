"""Configuration management for the AI Osrodek MLOps pipeline."""

from logging import getLogger
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from core.utils.logging import setup_logging

setup_logging()
logger = getLogger(__name__)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations."""
    if not configs:
        return OmegaConf.create({})

    merged = configs[0]
    for config in configs[1:]:
        merged = OmegaConf.merge(merged, config)

    return merged


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf config to regular dict."""
    return OmegaConf.to_container(config, resolve=True)


class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, configs_dir: str):
        self.configs_dir = Path(configs_dir) 
        self._configs = {}
    
    def load_config(self, config_name: str, config_dir: str) -> DictConfig:
        """Load configuration from file."""
        with initialize_config_dir(config_dir = str(self.configs_dir / config_dir), version_base="1.1"):
            config = compose(config_name=config_name)
            logger.info(f"Loaded config: {config}")
            self._configs[config_name] = config
            return config

    def get_available_configs(self):
        """Get configs available in the config directory.
        Return a mapping to all .yaml files in nested directories.
        """
        config_mapping = {}
        for dir in os.listdir(self.configs_dir):
            if os.path.isdir(self.configs_dir / dir):
                logger.info(f"Found config directory: {dir}")
        return config_mapping
    
    def get_config(self, config_name: str) -> Optional[DictConfig]:
        """Get cached configuration."""
        return self._configs.get(config_name)

