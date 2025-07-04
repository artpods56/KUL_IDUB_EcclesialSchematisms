# src/wikichurches/train.py
import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

import wandb
from dataset.filters import filter_schematisms, merge_filters
from dataset.maps import convert_to_grayscale, map_labels, merge_maps
from dataset.stats import compute_dataset_stats
from dataset.utils import get_dataset, load_labels, prepare_dataset
from dataset.parser import process_dataset_sample
from lmv3.utils.config import config_to_dict
from lmv3.utils.utils import get_device

from llm.evaluator import DatasetEvaluator
from shared import CONFIGS_DIR
from logger.setup import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

load_dotenv()



@hydra.main(config_path=str(CONFIGS_DIR / "llm"), config_name="evaluation_config", version_base=None)
def main(cfg: DictConfig) -> None:

    logger.info("Starting evaluation script...")
    logger.info(f"Using device: {get_device(cfg)}")
    
    # --- Load configuration ---
    run = None
    if cfg.wandb.enable:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.wandb.name}-{datetime.now().isoformat()}",
            tags=cfg.wandb.tags,
            config=config_to_dict(cfg),
        )

    # --- Prepare dataset ---
    dataset = get_dataset(cfg)
    
    logger.info("Raw dataset stats:")
    logger.info(json.dumps(
                compute_dataset_stats(dataset), indent=4, ensure_ascii=False)
                    )

    filters = [
        filter_schematisms(cfg.dataset.schematisms_to_train),
    ]
    maps = [
        map_labels(cfg.dataset.classes_to_remove), 
        convert_to_grayscale
        ]
    
    dataset = dataset.filter(merge_filters(filters), num_proc=8)
    dataset = dataset.map(merge_maps(maps), num_proc=8)

    logger.info("Cleaned dataset stats:")
    logger.info(json.dumps(
                compute_dataset_stats(dataset), indent=4, ensure_ascii=False)
                    )
    

    id2label, label2id, sorted_classes = load_labels(dataset)
    num_labels = len(sorted_classes)
    label_list = [id2label[i] for i in range(len(id2label))]


    # --- Get dataset splits ---
    dataset = dataset.shuffle(seed=42)
    train_val = dataset.train_test_split(
        test_size=cfg.dataset.test_size, seed=cfg.dataset.seed
    )
    test_val = train_val["test"].train_test_split(test_size=0.5, seed=cfg.dataset.seed)
    final_dataset = {
        "train": train_val["train"],
        "validation": test_val["train"],
        "test": test_val["test"],
    }

    logger.info(f"Final lengths  of dataset splits: "
                f"train={len(final_dataset['train'])}, "
                f"validation={len(final_dataset['validation'])}, "
                f"test={len(final_dataset['test'])}")

    
    logger.info("Starting evaluation...")

    
    evaluator = DatasetEvaluator(
            config=cfg
        )
    evaluator.evaluate_dataset(
        final_dataset["test"],
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
