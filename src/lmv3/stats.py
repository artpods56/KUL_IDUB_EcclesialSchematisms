from typing import Dict, Iterable
from collections import Counter
import re

PAGE_ONLY = {"B-page_number", "I-page_number", "O"}

def compute_dataset_stats(dataset) -> Dict:
    counter_template = {
        "positive": 0,
        "negative": 0,
        "page_number_only": 0,
    }

    schematisms = {}

    for ex in dataset:
        full_filename = ex["image"]
        splits = full_filename.split("_")
        filename = splits.pop()
        schematism = "_".join(splits)

        if schematism not in schematisms:
            schematisms[schematism] = counter_template.copy()

        lbl_set = set(ex["labels"])  # ← już stringi, więc nic nie mapujemy

        if lbl_set - {"O"}:
            schematisms[schematism]["positive"] += 1
        else:
            schematisms[schematism]["negative"] += 1

        if lbl_set <= PAGE_ONLY:
            schematisms[schematism]["page_number_only"] += 1

    total = len(dataset)
    stats = {
        "total_examples": total,
        "positive": sum(s["positive"] for s in schematisms.values()),
        "negative": sum(s["negative"] for s in schematisms.values()),
        "neg_to_pos_ratio": (
            sum(s["negative"] for s in schematisms.values())
            / sum(s["positive"] for s in schematisms.values())
            if sum(s["positive"] for s in schematisms.values())
            else None
        ),
        "page_number_only": sum(s["page_number_only"] for s in schematisms.values()),
        "page_number_only_ratio": (
            sum(s["page_number_only"] for s in schematisms.values()) / total
            if total
            else None
        ),
        "schematisms_stats": schematisms,
    }

    return stats