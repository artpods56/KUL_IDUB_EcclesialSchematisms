from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger

type Logger = Any


def get_logger(name: Any) -> BoundLogger:
    return cast(BoundLogger, structlog.get_logger(name))

