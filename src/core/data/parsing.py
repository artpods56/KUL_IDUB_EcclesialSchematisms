"""
BIO Parsing Utilities for Ecclesiastical Schematisms

This module provides utilities to convert BIO-tagged annotations into structured JSON
format for ecclesiastical schematism documents.
"""

import json
from typing import List, Dict, Any, Optional, Tuple
import operator
from thefuzz import fuzz
import re, unicodedata
import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely import wkt
from dotenv import load_dotenv


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
            if ent_type is None:
                # Treat as B- (start a new entity, or skip, or log warning)
                buff, ent_type = [w], t
            else:
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

def repair_bio_labels(labels):
    repaired = []
    prev_type = None
    for i, tag in enumerate(labels):
        if tag.startswith("B-"):
            _, curr_type = tag.split("-", 1)
            if prev_type == curr_type:
                # Consecutive B- of same type: convert to I-
                repaired.append(f"I-{curr_type}")
            else:
                repaired.append(tag)
            prev_type = curr_type
        elif tag.startswith("I-"):
            _, curr_type = tag.split("-", 1)
            # If previous type is not the same, or None, treat as B-
            if prev_type != curr_type or prev_type is None:
                repaired.append(f"B-{curr_type}")
            else:
                repaired.append(tag)
            prev_type = curr_type
        else:
            repaired.append(tag)
            prev_type = None
    return repaired

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
    # if bboxes:
        # words, labels = sort_by_layout(words, bboxes, labels)
    spans = bio_to_spans(words, labels)

    # Initialize result structure
    page_number = None
    entries = []
    deanery = None

    # Running buffer for each parish block
    current = {"deanery": None, "parish": None, "dedication": None, "building_material": None}

    for ent_type, text in spans:
        if ent_type == "page_number":
            page_number = text
        elif ent_type == "parish":
            # Start a new entry - flush previous if it exists
            if current["parish"]:
                entries.append(current)
                current = {"deanery": deanery ,"parish": None, "dedication": None, "building_material": None}
            current["parish"] = text
        elif ent_type == "deanery":
            deanery = text
            current["deanery"] = deanery
        elif ent_type == "dedication":
            current["dedication"] = text
        elif ent_type == "building_material":
            current["building_material"] = text



    # Flush last entry if it exists
    if current["parish"]:
        entries.append(current)

    return {
        "page_number": page_number,
        "entries": entries,
    }
