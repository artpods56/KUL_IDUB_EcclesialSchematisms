# src/wikichurches/train.py

import numpy as np
import argparse
import json
import os
from datetime import datetime

import hydra
import wandb
from omegaconf import OmegaConf

import gc

from datasets import load_dataset, DownloadMode
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    LayoutLMv3ForTokenClassification,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import default_data_collator

from metrics import build_compute_metrics
from trainers import FocalLossTrainer
from utils import load_labels, prepare_dataset, log_predictions_to_wandb, get_device
from stats import compute_dataset_stats
from filters import merge_filters, filter_schematisms
from maps import merge_maps, map_labels, convert_to_grayscale
from inference import retrieve_predictions

from dotenv import load_dotenv
import os

load_dotenv() 


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    primitive = OmegaConf.to_container(cfg, resolve=True)

    device = get_device(cfg)
    print(f"Using device: {device}")

    if cfg.wandb.enable:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=f"{cfg.wandb.name}-{datetime.now().isoformat()}",
            tags=cfg.wandb.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )



    dataset = load_dataset(
        cfg.dataset.name,
        "default",
        split="train",
        token="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",
        trust_remote_code=True,
        num_proc=8,
        download_mode=DownloadMode.FORCE_REDOWNLOAD if cfg.dataset.force_download else DownloadMode.REUSE_CACHE_IF_EXISTS,
    )

    raw_stats = compute_dataset_stats(dataset)
    print("Raw dataset stats:")
    print(json.dumps(raw_stats, indent=4, ensure_ascii=False))

    filters = [
        filter_schematisms(cfg.dataset.schematisms_to_train),
    ]

    dataset = dataset.filter(merge_filters(filters), num_proc=8)

    maps = [
        map_labels(cfg.dataset.classes_to_remove),
        convert_to_grayscale
    ]


    # dataset = dataset.map(merge_maps(maps), num_proc=8)

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
    
    train_val = dataset.train_test_split(test_size=cfg.dataset.test_size, seed=cfg.dataset.seed)
    test_val  = train_val["test"].train_test_split(test_size=0.5, seed=cfg.dataset.seed)


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
    
    import torch                      # already computed
    alpha = torch.ones(num_labels, dtype=torch.float32)
    alpha[label2id["O"]] = 0.05

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(
            label_list, return_entity_level_metrics=cfg.metrics.return_entity_level_metrics
        ),
        focal_loss_alpha=alpha,
        focal_loss_gamma=cfg.focal_loss.gamma,
        task_type= "multi-class",
        num_classes=len(id2label)
    )





    trainer.train()


    validation_results = trainer.evaluate(eval_dataset=eval_dataset)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    if cfg.wandb.enable and cfg.wandb.log_predictions:
        log_predictions_to_wandb(
            model=model,
            processor=processor,
            dataset=test_dataset,
            id2label=id2label,
            label2id=label2id,
            num_samples=cfg.wandb.num_prediction_samples,
        )
        
    #     log_predictions_to_wandb(
    #         model=model,
    #         processor=processor,
    #         dataset=final_dataset["test"],
    #         id2label=id2label,
    #         label2id=label2id,
    #         num_samples=cfg.wandb.num_prediction_samples,
    #     )

    

    if cfg.wandb.enable:
        wandb.log({"validation_results": validation_results})
        wandb.log({"test_results": test_results})
        wandb.log({"raw_stats": raw_stats})
        wandb.log({"training_stats": training_stats})

        run.finish()

if __name__ == "__main__":
    main()
