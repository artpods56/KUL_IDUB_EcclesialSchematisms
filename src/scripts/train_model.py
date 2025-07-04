#!/usr/bin/env python3
"""CLI script for training models in the AI Osrodek pipeline."""

import argparse
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config_manager
from core.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train models for document processing")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lmv3", "donut", "llm"],
        help="Model type to train"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Name for the training experiment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without actual training"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info(f"Starting training for model: {args.model}")
    
    try:
        # Load configuration
        config = config_manager.load_config("base", args.config)
        logger.info(f"Loaded configuration from: {args.config}")
        
        if args.dry_run:
            logger.info("DRY RUN: Would start training with the following configuration:")
            logger.info(config_manager.config_to_dict(config))
            return
        
        # Import and run the appropriate training script
        if args.model == "lmv3":
            from core.models.lmv3 import main as train_lmv3
            train_lmv3()
        elif args.model == "donut":
            logger.error("Donut training not yet implemented in new structure")
            sys.exit(1)
        elif args.model == "llm":
            logger.error("LLM training not yet implemented in new structure")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
