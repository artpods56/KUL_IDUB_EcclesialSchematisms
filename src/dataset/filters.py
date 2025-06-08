from typing import List, Callable

def merge_filters(filters: List[Callable]) -> Callable:
    """
    Merges multiple filter functions into a single function.
    Each filter function should return True for examples to keep.
    """
    def merged_filter(example):
        return all(f(example) for f in filters)
    
    return merged_filter

def filter_schematisms(schematisms_to_train: set) -> Callable:
    def _filter_fn(example):
        full_filename = example["image"]
        splits = full_filename.split("_")
        filename = splits.pop()
        schematism = "_".join(splits)
        return schematism in schematisms_to_train
    return _filter_fn


    


