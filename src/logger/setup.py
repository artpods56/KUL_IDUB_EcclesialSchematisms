import logging.config
import os
from pathlib import Path

import yaml

from shared import CONFIGS_DIR, PROJECT_ROOT

def prepare_logs_dir(logs_dir: Path):
    """
    Prepare the logs directory by creating it if it doesn't exist.
    """
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created logs directory: {logs_dir}")
    else:
        print(f"Logs directory already exists: {logs_dir}")

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
        
    prepare_logs_dir(Path(logs_dir.parent))
    logging.config.dictConfig(config)
