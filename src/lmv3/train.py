# src/wikichurches/train.py
import json
import os
from datetime import datetime
from typing import cast

import hydra
from datasets import Dataset, DownloadMode, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from stats import compute_dataset_stats
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from transformers.data.data_collator import default_data_collator
from transformers.training_args import TrainingArguments

import wandb
from lmv3.filters import filter_schematisms, merge_filters
from lmv3.maps import convert_to_grayscale, map_labels, merge_maps
from lmv3.metrics import build_compute_metrics
from lmv3.trainers import FocalLossTrainer
from lmv3.utils.config import config_to_dict
from lmv3.utils.utils import get_device, load_labels, prepare_dataset

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

    id2label, label2id, sorted_classes = load_labels(dataset)
    num_labels = len(sorted_classes)

    label_list = [id2label[i] for i in range(len(id2label))]

    processor = AutoProcessor.from_pretrained(cfg.processor.checkpoint, apply_ocr=False)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        cfg.model.checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
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

    train_dataset = prepare_dataset(
        final_dataset["train"], processor, id2label, label2id, dataset_config
    )
    eval_dataset = prepare_dataset(
        final_dataset["validation"], processor, id2label, label2id, dataset_config
    )
    test_dataset = prepare_dataset(
        final_dataset["test"], processor, id2label, label2id, dataset_config
    )

    print(f"Train dataset size: {len(final_dataset['train'])}")
    print(f"Validation dataset size: {len(final_dataset['validation'])}")
    print(f"Test dataset size: {len(final_dataset['test'])}")

    training_args = TrainingArguments(**cfg.training)

    num_labels = len(id2label)

    import torch  # already computed

    alpha = torch.ones(num_labels, dtype=torch.float32)
    alpha[label2id["O"]] = 0.05

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(
            label_list,
            return_entity_level_metrics=cfg.metrics.return_entity_level_metrics,
        ),
        focal_loss_alpha=alpha,
        focal_loss_gamma=cfg.focal_loss.gamma,
        task_type="multi-class",
        num_classes=len(id2label),
    )

    trainer.train()

    # validation_results = trainer.evaluate(eval_dataset=eval_dataset)
    # test_results = trainer.evaluate(eval_dataset=test_dataset)

    # if cfg.wandb.enable and cfg.wandb.log_predictions:
    #     log_predictions_to_wandb(
    #         model=model,
    #         processor=processor,
    #         dataset=test_dataset,
    #         id2label=id2label,
    #         label2id=label2id,
    #         num_samples=cfg.wandb.num_prediction_samples,
    #     )

    #     log_predictions_to_wandb(
    #         model=model,
    #         processor=processor,
    #         dataset=final_dataset["test"],
    #         id2label=id2label,
    #         label2id=label2id,
    #         num_samples=cfg.wandb.num_prediction_samples,
    #     )

    if cfg.wandb.enable:

        run.finish()


if __name__ == "__main__":
    main()
