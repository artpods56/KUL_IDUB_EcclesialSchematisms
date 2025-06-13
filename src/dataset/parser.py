"""
BIO Parsing Utilities for Ecclesiastical Schematisms

This module provides utilities to convert BIO-tagged annotations into structured JSON
format for ecclesiastical schematism documents.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
import operator


def bio_to_spans(words: List[str], labels: List[str]) -> List[Tuple[str, str]]:
    """
    Convert parallel `words`, `labels` (BIO) into a list of
    (entity_type, "concatenated text") tuples.
    
    Args:
        words: List of word tokens
        labels: List of BIO labels corresponding to words
        
    Returns:
        List of (entity_type, text) tuples
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


def sort_by_layout(words: List[str], bboxes: List[List[int]], labels: List[str]) -> Tuple[List[str], List[str]]:
    """
    Sort words in reading order (top-to-bottom, left-to-right) using bounding boxes.
    
    Args:
        words: List of word tokens
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of BIO labels
        
    Returns:
        Tuple of (sorted_words, sorted_labels)
    """
    if not bboxes or len(bboxes) != len(words):
        return words, labels
        
    # Sort by y-coordinate first (top), then x-coordinate (left)
    order = sorted(range(len(words)),
                   key=lambda i: (bboxes[i][1], bboxes[i][0]))  # y, then x
    return [words[i] for i in order], [labels[i] for i in order]


def build_page_json(words: List[str], bboxes: List[List[int]], labels: List[str]) -> Dict[str, Any]:
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
    
    Args:
        words: List of word tokens
        bboxes: List of bounding boxes [x1, y1, x2, y2]
        labels: List of BIO labels
        
    Returns:
        Dictionary with parsed JSON structure
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


def process_dataset_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single dataset sample and convert it to JSON format.
    
    Args:
        sample: Dictionary with keys 'words', 'bboxes', 'labels'
    
    Returns:
        Dictionary with parsed JSON structure
    """
    return build_page_json(
        words=sample["words"],
        bboxes=sample.get("bboxes", []), 
        labels=sample["labels"]
    )


def batch_process_dataset(dataset, output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Process multiple dataset samples and optionally save to file.
    
    Args:
        dataset: Dataset or list of samples
        output_file: Optional path to save JSON output
        
    Returns:
        List of parsed JSON structures
    """
    results = []
    
    for sample in dataset:
        try:
            parsed = process_dataset_sample(sample)
            results.append({
                "image": sample.get("image", sample.get("image_path", "unknown")),
                "parsed_data": parsed
            })
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} parsed samples to {output_file}")
    
    return results


def validate_bio_tags(labels: List[str]) -> List[str]:
    """
    Validate and report issues with BIO tag sequences.
    
    Args:
        labels: List of BIO labels
        
    Returns:
        List of validation issues found
    """
    issues = []
    
    for i, label in enumerate(labels):
        if label == "O":
            continue
            
        if "-" not in label:
            issues.append(f"Position {i}: Invalid tag format '{label}' (missing BIO prefix)")
            continue
            
        prefix, entity_type = label.split("-", 1)
        
        if prefix not in ["B", "I"]:
            issues.append(f"Position {i}: Invalid BIO prefix '{prefix}' in '{label}'")
            
        if prefix == "I" and i > 0:
            prev_label = labels[i-1]
            if prev_label == "O":
                issues.append(f"Position {i}: I-tag '{label}' follows O tag")
            elif "-" in prev_label:
                prev_entity = prev_label.split("-", 1)[1]
                if prev_entity != entity_type:
                    issues.append(f"Position {i}: I-tag '{label}' doesn't match previous entity '{prev_entity}'")
    
    return issues


# Example usage and testing functions
def demo_parsing():
    """Demonstrate the BIO parsing functionality with sample data."""
    
    # Sample data
    sample_words = ["41", "Czermin", "P.", "E.", "p.", "mur.", "S.", "Clementem", "P.", "M.", "et", "S.", "Annam"]
    sample_labels = ["B-page_number", "B-parish", "O", "O", "O", "B-building_material", "B-dedication", "I-dedication", "I-dedication", "I-dedication", "I-dedication", "I-dedication", "I-dedication"]
    sample_bboxes = [[10, 10, 20, 20] for _ in sample_words]  # Mock bboxes
    
    print("=== BIO Parsing Demo ===")
    print(f"Words: {sample_words}")
    print(f"Labels: {sample_labels}")
    print()
    
    # Test BIO validation
    issues = validate_bio_tags(sample_labels)
    if issues:
        print("BIO Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ BIO tags are valid")
    print()
    
    # Test span extraction
    spans = bio_to_spans(sample_words, sample_labels)
    print("Extracted spans:")
    for entity_type, text in spans:
        print(f"  {entity_type}: '{text}'")
    print()
    
    # Test JSON structure building
    result = build_page_json(sample_words, sample_bboxes, sample_labels)
    print("Final JSON structure:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    demo_parsing()