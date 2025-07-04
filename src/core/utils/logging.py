import logging.config
import os
from pathlib import Path

import yaml
import structlog

from core.utils.shared import CONFIGS_DIR, REPOSITORY_ROOT


def setup_logging(env_key: str = "LOGGING_CONFIG_PATH"):
    """
    Set up logging from YAML config file.
    Fallbacks to default path if env var is not set.
    Ensures log directories exist.
    """
    default_path = CONFIGS_DIR / "logging" / "config.yaml"
    cfg_path = Path(os.getenv(env_key, default_path))

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Logging config not found: {cfg_path}")

    with cfg_path.open("r") as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as e:
            raise RuntimeError(f"Failed to parse YAML logging config: {e}")

    for name, handler in config.get("handlers", {}).items():
        if "filename" in handler:
            path = Path(handler["filename"])
            if not path.is_absolute():
                path = REPOSITORY_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            handler["filename"] = str(path)

    logging.config.dictConfig(config)

    # Configure structlog so that `structlog.get_logger()` plays nicely with the
    # standard library's logging module that we just initialised above.
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")

    structlog.configure(
        processors=[
            # Drop messages that are below the set level early for efficiency.
            structlog.stdlib.filter_by_level,
            # Add useful metadata.
            timestamper,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            # Render stack-/tracebacks if present.
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Render the final structured log as JSON so that it becomes the
            # log message handled by the standard library logging handlers
            # defined in our YAML config (rich handler, file handlers, …).
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLogger().level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()
