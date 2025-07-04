import logging.config
import os
from pathlib import Path

import yaml
from core.utils.shared import CONFIGS_DIR, PROJECT_ROOT

def setup_logging(env_key: str = "LOGGING_CONFIG_PATH"):
    print(CONFIGS_DIR)
    default_path = CONFIGS_DIR / "logging" / "config.yaml"

    cfg_path = Path(os.getenv(env_key, default_path))
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Logging config not found: {cfg_path}")

    with cfg_path.open("r") as fp:
        config = yaml.safe_load(fp.read())

    logs_dir = config["handlers"]["file"]["filename"]
    
    if not logs_dir.startswith("/"):
        logs_dir = PROJECT_ROOT / logs_dir  # Make it absolute if it's relative
        config["handlers"]["file"]["filename"] = str(logs_dir)
    
    if not logs_dir.parent.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
        
    logging.config.dictConfig(config)
