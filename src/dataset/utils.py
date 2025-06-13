import os
from typing import Dict, List, cast

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


def get_dataset(cfg: DictConfig) -> Dataset:

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

    return dataset



def parse_to_json(words, labels, boxes):
    """
    Convert words, labels, and boxes to a JSON-like format.
    """
    # Use the existing bio_to_spans function or implement it here
    spans = bio_to_spans(words, labels)
    
    # Build the JSON structure
    result = build_page_json(words, boxes, labels)
    return result


def bio_to_spans(words: List[str], labels: List[str]) -> List[tuple]:
    """
    Convert parallel `words`, `labels` (BIO) into a list of
    (entity_type, "concatenated text") tuples.
    """
    spans = []
    buff, ent_type = [], None

    for w, tag in zip(words, labels):
        if tag == "O":
            if buff:
                spans.append((ent_type, " ".join(buff)))
                buff, ent_type = [], None
            continue

        if "-" not in tag:  # Handle cases where tag doesn't have BIO prefix
            continue
            
        prefix, t = tag.split("-", 1)
        if prefix == "B" or (ent_type and t != ent_type):
            if buff:
                spans.append((ent_type, " ".join(buff)))
            buff, ent_type = [w], t
        else:  # "I"
            buff.append(w)

    if buff:
        spans.append((ent_type, " ".join(buff)))
    return spans


def sort_by_layout(words, bboxes, labels):
    """Sort words in reading order (top-to-bottom, left-to-right)"""
    if not bboxes or len(bboxes) != len(words):
        return words, labels
        
    # Sort by y-coordinate first (top), then x-coordinate (left)
    order = sorted(range(len(words)),
                   key=lambda i: (bboxes[i][1], bboxes[i][0]))  # y, then x
    return [words[i] for i in order], [labels[i] for i in order]


def build_page_json(words, bboxes, labels):
    """
    Build the target JSON structure from BIO-tagged annotations.
    
    Expected output format:
    {
      "page_number": "<string | null>",
      "deanery": "<string | null>", 
      "entries": [
        {
          "parish": "<string>",
          "dedication": "<string>", 
          "building_material": "<string>"
        },
        ...
      ]
    }
    """
    # Sort words in reading order for better parsing
    if bboxes:
        words, labels = sort_by_layout(words, bboxes, labels)
    
    spans = bio_to_spans(words, labels)

    # Initialize result structure
    page_number = None
    deanery = None
    entries = []

    # Running buffer for each parish block
    current = {"parish": None, "dedication": None, "building_material": None}

    for ent_type, text in spans:
        if ent_type == "page_number":
            page_number = text
        elif ent_type == "deanery":
            deanery = text
        elif ent_type == "parish":
            # Start a new entry - flush previous if it exists
            if current["parish"]:
                entries.append(current)
                current = {"parish": None, "dedication": None, "building_material": None}
            current["parish"] = text
        elif ent_type == "dedication":
            current["dedication"] = text
        elif ent_type == "building_material":
            current["building_material"] = text

    # Flush last entry if it exists
    if current["parish"]:
        entries.append(current)

    return {
        "page_number": page_number,
        "deanery": deanery,
        "entries": entries,
    }


def process_dataset_sample(sample):
    """
    Process a single dataset sample and convert it to JSON format.
    
    Args:
        sample: Dictionary with keys 'words', 'bboxes', 'labels'
    
    Returns:
        Dictionary with parsed JSON structure
    """
    return build_page_json(
        words=sample["words"],
        bboxes=sample["bboxes"], 
        labels=sample["labels"]
    )
