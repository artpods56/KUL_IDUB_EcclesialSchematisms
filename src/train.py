# src/wikichurches/train.py
import argparse
import json
import os
from datetime import datetime

import hydra
import wandb
from omegaconf import OmegaConf

from datasets import load_dataset
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
from utils import load_labels, prepare_dataset

from dotenv import load_dotenv
import os

load_dotenv() 



@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    primitive = OmegaConf.to_container(cfg, resolve=True)

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
    )

    def is_truly_positive(example):
        return all(label == "O" for label in example["labels"])

    #dataset = dataset.filter(is_truly_positive)
    
    id2label, label2id, sorted_classes = load_labels(dataset)
    num_labels = len(sorted_classes)
    

    print(id2label)

    print(label2id)
    print(num_labels)
    print(len(label2id.keys()))

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
    
    train_valtest = dataset.train_test_split(test_size=0.2, seed=cfg.dataset.seed)
    val_test = train_valtest["test"].train_test_split(test_size=0.5, seed=cfg.dataset.seed)

    final_dataset = {
        "train": train_valtest["train"], 
        "validation": val_test["train"], 
        "test": val_test["test"]              
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

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=build_compute_metrics(
            label_list, return_entity_level_metrics=cfg.metrics.return_entity_level_metrics
        ),
        focal_loss_alpha=cfg.focal_loss.alpha,
        focal_loss_gamma=cfg.focal_loss.gamma,
    )

    trainer.train()

    if cfg.wandb.enable:
        run.finish()

if __name__ == "__main__":
    main()
