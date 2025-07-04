import json
import os
from datetime import datetime
from typing import cast

import hydra
from datasets import Dataset, DownloadMode, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from dataset.stats import compute_dataset_stats # Updated import
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

import wandb
from dataset.filters import filter_schematisms, merge_filters # Updated import
from dataset.maps import convert_to_grayscale, map_labels, merge_maps # Updated import
from lmv3.utils.config import config_to_dict
from lmv3.utils.inference_utils import get_model_and_processor
# Updated imports for load_labels and prepare_dataset, get_device remains
from lmv3.utils.utils import get_device
from dataset.utils import load_labels, prepare_dataset
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

    download_mode = (
        DownloadMode.FORCE_REDOWNLOAD
        if cfg.dataset.force_download
        else DownloadMode.REUSE_CACHE_IF_EXISTS
    )

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("Huggingface token is missing.")

    dataset = cast(
        Dataset,
        load_dataset(
            path=cfg.dataset.path,
            name=cfg.dataset.name,
            split=cfg.dataset.split,
            token=os.getenv("HF_TOKEN"),
            trust_remote_code=cfg.dataset.trust_remote_code,
            num_proc=cfg.dataset.num_proc,
            download_mode=download_mode,
            keep_in_memory=cfg.dataset.keep_in_memory,
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

    model, processor = get_model_and_processor(cfg)

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

        samples_to_log = log_predictions_to_wandb(
            model=model,
            processor=processor,
            datasets_splits=[ # This argument name is different from the one in lmv3.utils.utils.log_predictions_to_wandb
                final_dataset["validation"],
                final_dataset["test"],
            ],
            config=cfg, # This implies id2label might be inside cfg or handled by the processor/model internally
        )

        run.log({"eval_samples": samples_to_log})

        run.finish()


if __name__ == "__main__":
    main()
