from typing import List, Callable
from PIL import Image

def merge_maps(mappers: List[Callable]) -> Callable:
    """
    Merges multiple `map`-like functions into a single callable.
    Each function should accept a single example and return a modified example.
    """
    def merged_map(example):
        for fn in mappers:
            example = fn(example)
        return example
    return merged_map

def map_labels(classes_to_remove: set) -> Callable:
    def _map_fn(example):
        example["labels"] = [
            label if label not in classes_to_remove else "O"
            for label in example["labels"]
        ]
        return example
    return _map_fn

def convert_to_grayscale(example):
    img = example["image_pil"].convert("L")         # grayscale (1 kanał)
    example["image_pil"] = img.convert("RGB")       # wracamy do 3 kanałów
    return example
