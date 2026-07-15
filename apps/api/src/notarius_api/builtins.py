from notarius_core.operators.arithmetic import ARITHMETIC
from notarius_core.operators.sequences import SEQUENCES
from notarius_core.operators.sources import SOURCES
from notarius_core.operators.tables import TABLES
from notarius_core.operators.text import TEXT
from notarius_core.plugins import Plugin


def builtin_plugins() -> tuple[Plugin, ...]:
    """Plugins shipped with the Notarius host and installed explicitly."""

    return (
        SOURCES,
        SEQUENCES,
        ARITHMETIC,
        TEXT,
        TABLES,
    )
