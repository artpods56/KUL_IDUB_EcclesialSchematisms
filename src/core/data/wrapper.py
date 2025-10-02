

from typing import Iterable
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict
from core.data.parsing import build_page_json, repair_bio_labels


class DatasetWrapper(Iterable):
    def __init__(self, dataset: Dataset | DatasetDict | IterableDataset | IterableDatasetDict):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset[idx]

        parts = row["image"].split("_")
        if len(parts) >= 2:
            schematism_name = '_'.join(parts[:-1])  # Everything except the last part
            filename = parts[-1]
        else:
            raise ValueError(f"Invalid filename")
        #get page_json

        words, bboxes, labels = row["words"], row["bboxes"], row["labels"]
        repaired_labels = repair_bio_labels(labels)

        image_pil = row["image_pil"]
        page_json = build_page_json(words=words, bboxes=bboxes, labels=repaired_labels)

        return {
            "schematism_name": schematism_name,
            "image_filename": filename,
            "image_pil": image_pil,
            "page_json": page_json}

    def __repr__(self):
        return f"""DatasetWrapper(
        {self.dataset}
        """

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __next__(self):
        return self.__iter__()
    
    def __getattr__(self, name):
        return getattr(self.dataset, name)
    
    def filter(self, filter_fn):
        return DatasetWrapper(self.dataset.filter(filter_fn))
    
    def map(self, map_fn):
        return DatasetWrapper(self.dataset.map(map_fn))
    
    def batch(self, batch_size):
        return DatasetWrapper(self.dataset.batch(batch_size))

    