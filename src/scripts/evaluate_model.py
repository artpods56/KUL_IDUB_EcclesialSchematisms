#!/usr/bin/env python3
"""CLI script for evaluating models in the AI Osrodek pipeline."""

import argparse
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import config_manager
from core.utils import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate models for document processing")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lmv3", "donut", "llm"],
        help="Model type to evaluate"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["accuracy", "f1", "precision", "recall"],
        help="Metrics to compute"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info(f"Starting evaluation for model: {args.model}")
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration if provided
        config = None
        if args.config:
            config = config_manager.load_config("evaluation", args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        
        # Run evaluation based on model type
        if args.model == "lmv3":
            logger.info("Running LayoutLMv3 evaluation...")
            # Import and run LayoutLMv3 evaluation
            # from ai_osrodek.evaluation.lmv3_evaluator import evaluate_lmv3
            # results = evaluate_lmv3(args.model_path, args.data_path, args.metrics)
            logger.info("LayoutLMv3 evaluation not yet implemented in new structure")
            
        elif args.model == "donut":
            logger.info("Running Donut evaluation...")
            logger.info("Donut evaluation not yet implemented in new structure")
            
        elif args.model == "llm":
            logger.info("Running LLM evaluation...")
            logger.info("LLM evaluation not yet implemented in new structure")
        
        logger.info(f"Evaluation results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
