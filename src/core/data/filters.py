from typing import Any, Iterable, List, Callable, Union

import json

def merge_filters(filters: List[Callable]) -> Callable:
    """
    Merges multiple filter functions into a single function.
    Each filter function should return True for examples to keep.
    """
    def merged_filter(example):
        return all(f(example) for f in filters)
    
    return merged_filter

def filter_schematisms(to_filter: str | list[str]):
    """Filter schematisms by schematism description or list of schematisms."""
    def _filter_fn(schematism_name: str) -> bool:
        if isinstance(to_filter, str):
            return schematism_name == to_filter
        else:
            return schematism_name in to_filter
    return _filter_fn

def filter_empty_samples(results: str | dict[str, list[Any]]):
    """Filter out empty examples (i.e. with empty labels).
    
    results = '{"page_number": null, "entries": []}'
    """
    if isinstance(results, str):
        try:
            parsed: dict[str, Any] = json.loads(results)
        except json.JSONDecodeError:
            return False
    else:
        parsed = results

    entries = parsed.get("entries", [])
    return bool(entries)



    


