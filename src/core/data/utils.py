import os
from typing import Dict, List, cast, Union

from datasets import (
    Array2D,
    Array3D,
    Dataset,
    DownloadMode,
    Features,
    Sequence,
    Value,
    load_dataset,
)
from omegaconf import DictConfig

from data.wrapper import DatasetWrapper

def load_labels(dataset: Dataset):
    classes = []
    for example in dataset:
        if "labels" in example.keys():
            if isinstance(example["labels"], list):
                classes.extend(example["labels"])
            else:
                classes.append(example["labels"])

    unique_classes = set(classes)
    sorted_classes = sorted(list(unique_classes))

    id2label = {i: label for i, label in enumerate(sorted_classes)}
    label2id = {label: i for i, label in enumerate(sorted_classes)}
    return id2label, label2id, sorted_classes


def prepare_dataset(dataset: Dataset, processor, id2label, label2id, dataset_config):

    def prepare_examples(examples):
        images = examples[dataset_config["image_column_name"]]
        words = examples[dataset_config["text_column_name"]]
        boxes = examples[dataset_config["boxes_column_name"]]
        word_labels = examples[dataset_config["label_column_name"]]

        # Since your dataset has string labels, always convert them to IDs
        label_ids = [[label2id[label] for label in seq] for seq in word_labels]

        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=label_ids,
            truncation=True,
            stride=128,
            padding="max_length",
            max_length=512,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.pop("offset_mapping")
        overflow_to_sample_mapping = encoding.pop("overflow_to_sample_mapping")
        return encoding

    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "attention_mask": Sequence(Value("int64")),
            "labels": Sequence(Value("int64")),
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
        }
    )

    prepared_dataset = dataset.map(
        prepare_examples,
        batched=True,
        remove_columns=dataset.column_names,
        features=features,
    )

    prepared_dataset.set_format("torch")

    return prepared_dataset


def _to_fractional(box: List[int]) -> Dict[str, float]:
    """
    LayoutLM* boxes are in the 0-1000 coordinate system.
    WandB defaults to ‘fractional’ domain = values in [0,1].
    """
    min_x, min_y, max_x, max_y = [v / 1000.0 for v in box]

    return {"minX": min_x, "maxX": max_x, "minY": min_y, "maxY": max_y}


def get_dataset(cfg: DictConfig, wrapper: bool = False) -> Union[Dataset, DatasetWrapper]:

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
            token=HF_TOKEN,
            trust_remote_code=cfg.dataset.trust_remote_code,
            num_proc=cfg.dataset.num_proc if cfg.dataset.num_proc > 0 else None,
            download_mode=download_mode,
            keep_in_memory=cfg.dataset.keep_in_memory,
            streaming=cfg.dataset.streaming,
        ),
    )

    if wrapper:
        return DatasetWrapper(dataset)
    else:
        return dataset


