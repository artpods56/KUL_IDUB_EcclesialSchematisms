from core.utils.shared import CONFIGS_DIR
from core.config.manager import ConfigManager

from structlog import get_logger
from core.utils.logging import setup_logging
setup_logging()

logger = get_logger(__name__)

from dotenv import load_dotenv
load_dotenv()


def main():

    logger.info("Generating default configs...")
    logger.info(f"Using config directory: {CONFIGS_DIR}")
    config_manager = ConfigManager(CONFIGS_DIR)
    logger.info("Starting generation of default configs...")
    config_manager.generate_default_configs()
    logger.info("Default configs generation completed successfully!")

if __name__ == "__main__":
    main()
