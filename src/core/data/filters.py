from typing import Iterable, List, Callable, Union

def merge_filters(filters: List[Callable]) -> Callable:
    """
    Merges multiple filter functions into a single function.
    Each filter function should return True for examples to keep.
    """
    def merged_filter(example):
        return all(f(example) for f in filters)
    
    return merged_filter

def filter_schematisms(to_filter: Union[str, Iterable]):
    """Filter schematisms by schematism name or list of schematisms."""
    def _filter_fn(schematism_name):
        if isinstance(to_filter, str):
            return schematism_name == to_filter
        else:
            return schematism_name in to_filter
    return _filter_fn


    


