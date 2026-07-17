from notarius_core.operators.arithmetic import ARITHMETIC
from notarius_core.operators.images import IMAGES
from notarius_core.operators.modules import MODULES
from notarius_core.operators.prompts import PROMPTS
from notarius_core.operators.schemas import SCHEMAS
from notarius_core.operators.sequences import SEQUENCES
from notarius_core.operators.text import TEXT
from notarius_core.plugins import Plugin


def builtin_plugins() -> tuple[Plugin, ...]:
    """Plugins shipped with the Notarius host and installed explicitly."""

    return (
        IMAGES,
        MODULES,
        SEQUENCES,
        ARITHMETIC,
        TEXT,
        SCHEMAS,
        PROMPTS,
    )
