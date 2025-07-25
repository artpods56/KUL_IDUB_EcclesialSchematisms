import logging.config
import os
from pathlib import Path
from typing import Optional, Literal, get_args
import logging, sys, structlog
import yaml
import structlog

from core.utils.shared import CONFIGS_DIR, REPOSITORY_ROOT

LOG_LEVELS = Literal["INFO", "DEBUG", "INFO", "WARNING", "ERROR"]


def setup_logging() -> None:

    logs_dir = REPOSITORY_ROOT / "tmp" / "logs"
    if not Path(logs_dir).exists():
        logs_dir.mkdir(parents=True, exist_ok=True)


    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(logs_dir / "app.log", encoding="utf-8")

    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=True),
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
    ]

    console_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=pre_chain,
    ))

    file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=pre_chain,
    ))

    logging.basicConfig(
        level="INFO",
        handlers=[console_handler, file_handler],
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )



