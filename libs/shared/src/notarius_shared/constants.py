from pathlib import Path

from notarius_shared.utils.path_utils import find_repository_root

REPOSITORY_ROOT: Path = find_repository_root()
TMP_DIR = REPOSITORY_ROOT / "tmp"
CONFIGS_DIR = REPOSITORY_ROOT / "configs"
DATA_DIR = REPOSITORY_ROOT / "data"
MODELS_DIR = REPOSITORY_ROOT / "ml_models"
CACHES_DIR = REPOSITORY_ROOT / "caches"
INPUTS_DIR = DATA_DIR / "inputs"
OUTPUTS_DIR = DATA_DIR / "outputs"
MAPPINGS_DIR = DATA_DIR / "mappings"
PDF_SOURCE_DIR = INPUTS_DIR / "pdfs"

MAX_LLM_RETRIES = 10
