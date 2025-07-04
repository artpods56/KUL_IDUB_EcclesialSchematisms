from pathlib import Path

def find_repository_root(start=None) -> Path:
    """
    Finds the repository root by searching for the .git directory.
    """
    path = Path(start or __file__).resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").is_dir():
            return parent
    raise RuntimeError("Could not find project root.")

REPOSITORY_ROOT = find_repository_root()

TMP_DIR = REPOSITORY_ROOT / "tmp"

CONFIGS_DIR = REPOSITORY_ROOT / "configs"