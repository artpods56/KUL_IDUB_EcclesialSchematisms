# src/wikichurches/train.py
import json
import os
from datetime import datetime

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

import wandb
from dataset.filters import filter_schematisms, merge_filters  # Updated import
from dataset.maps import convert_to_grayscale, map_labels, merge_maps  # Updated import
from dataset.stats import compute_dataset_stats  # Updated import
from dataset.utils import get_dataset, load_labels, prepare_dataset
from lmv3.utils.config import config_to_dict
from lmv3.utils.utils import get_device

load_dotenv()


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    device = get_device(cfg)
    print(f"Using device: {device}")

    if cfg.wandb.enable:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.wandb.name}-{datetime.now().isoformat()}",
            tags=cfg.wandb.tags,
            config=config_to_dict(cfg),
        )

    dataset = get_dataset(cfg)

    raw_stats = compute_dataset_stats(dataset)
    print("Raw dataset stats:")
    print(json.dumps(raw_stats, indent=4, ensure_ascii=False))

    filters = [
        filter_schematisms(cfg.dataset.schematisms_to_train),
    ]

    dataset = dataset.filter(merge_filters(filters), num_proc=8)

    maps = [map_labels(cfg.dataset.classes_to_remove), convert_to_grayscale]

    dataset = dataset.map(merge_maps(maps), num_proc=8)

    training_stats = compute_dataset_stats(dataset)
    print("Train / Eval  datasets stats:")
    print(json.dumps(training_stats, indent=4, ensure_ascii=False))

    id2label, label2id, sorted_classes = load_labels(dataset)
    num_labels = len(sorted_classes)

    label_list = [id2label[i] for i in range(len(id2label))]

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

    print(f"Train dataset size: {len(final_dataset['train'])}")
    print(f"Validation dataset size: {len(final_dataset['validation'])}")
    print(f"Test dataset size: {len(final_dataset['test'])}")

    if cfg.wandb.enable:

        run.finish()


if __name__ == "__main__":
    main()
