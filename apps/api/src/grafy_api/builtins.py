from grafy_core.operators.arithmetic import ARITHMETIC
from grafy_core.operators.images import IMAGES
from grafy_core.operators.modules import MODULES
from grafy_core.operators.prompts import PROMPTS
from grafy_core.operators.schemas import SCHEMAS
from grafy_core.operators.sequences import SEQUENCES
from grafy_core.operators.tables import TABLES
from grafy_core.operators.text import TEXT
from grafy_core.plugins import Plugin


def builtin_plugins() -> tuple[Plugin, ...]:
    """Plugins shipped with the Grafy host and installed explicitly."""

    return (
        IMAGES,
        MODULES,
        SEQUENCES,
        ARITHMETIC,
        TEXT,
        SCHEMAS,
        PROMPTS,
        TABLES,
    )
