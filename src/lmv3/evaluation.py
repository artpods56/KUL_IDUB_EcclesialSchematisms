import json
from datetime import datetime
from typing import cast

import hydra
import torch
from datasets import Dataset, DownloadMode, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from stats import compute_dataset_stats
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

import wandb
from lmv3.filters import filter_schematisms, merge_filters
from lmv3.maps import convert_to_grayscale, map_labels, merge_maps
from lmv3.utils.config import config_to_dict
from lmv3.utils.utils import get_device, load_labels, prepare_dataset
from lmv3.utils.wandb_utils import log_predictions_to_wandb

load_dotenv()


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = get_device(cfg)
    print(f"Using device: {device}")

    if cfg.wandb.enable:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"inference-{cfg.wandb.name}-{datetime.now().isoformat()}",
            tags=cfg.wandb.tags,
            config=config_to_dict(cfg),
        )

    dataset = load_dataset(
        cfg.dataset.name,
        "default",
        split="train",
        token="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",
        trust_remote_code=True,
        num_proc=8,
        download_mode=(
            DownloadMode.FORCE_REDOWNLOAD
            if cfg.dataset.force_download
            else DownloadMode.REUSE_CACHE_IF_EXISTS
        ),
    )

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


    processor = AutoProcessor.from_pretrained(cfg.processor.checkpoint, apply_ocr=True)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        cfg.inference.checkpoint,
    )

    dataset = dataset.shuffle(seed=42)
    dataset_config = {
        "image_column_name": cfg.dataset.image_column_name,
        "text_column_name": cfg.dataset.text_column_name,
        "boxes_column_name": cfg.dataset.boxes_column_name,
        "label_column_name": cfg.dataset.label_column_name,
    }

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
        samples_to_log = log_predictions_to_wandb(
            model=model,
            processor=processor,
            datasets_splits=[
                final_dataset["validation"],
                final_dataset["test"],
            ],
            config=cfg,
        )

        run.log({"eval_samples": samples_to_log})

        run.finish()


if __name__ == "__main__":
    main()
