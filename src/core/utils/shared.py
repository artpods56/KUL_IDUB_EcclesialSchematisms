"""Shared utilities used across the project.

This module defines :pydata:`REPOSITORY_ROOT` and related paths in a way that
does **not** require the presence of a ``.git`` directory inside production
containers.  The logic is:

1.  If an environment variable ``PROJECT_ROOT`` (or the historical
    ``AI_OSRODEK_ROOT``) is set, use it.
2.  Otherwise, walk upwards from the current file looking for a *marker* file
    that always exists in the repository, namely ``pyproject.toml``.
3.  As a last resort (for local development clones) fall back to detecting the
    ``.git`` directory.

This approach keeps containers lightweight (no need to mount or COPY the full
``.git`` history) while still allowing source checkouts to work seamlessly on
developer machines.
"""

from pathlib import Path
import os


def _env_override() -> Path | None:
    """Return repository root from environment variable if set."""
    env_root = os.getenv("PROJECT_ROOT", None)
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.exists():
            return root
    return None


def _find_by_marker(start: Path) -> Path | None:
    """Look upwards for ``pyproject.toml`` as a repository marker."""
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def find_repository_root(start: str | Path | None = None) -> Path:
    """Robustly detect the project root directory.

    The search order is:
    1. `PROJECT_ROOT` env var.
    2. Upwards walking search for ``pyproject.toml``.
    """
    if env_root := _env_override():
        return env_root

    # 2. Marker search (works in containers where pyproject.toml is copied)
    path = Path(start or __file__).resolve()
    if marker_root := _find_by_marker(path):
        return marker_root

    raise RuntimeError("Could not find project root. Ensure project includes project.toml file or set PROJECT_ROOT environmental variable.")

REPOSITORY_ROOT: Path = find_repository_root()

TMP_DIR = REPOSITORY_ROOT / "tmp"

CONFIGS_DIR = REPOSITORY_ROOT / "configs"