"""Utilities for handling and saving generated mappings."""

import json
import csv
from pathlib import Path
import datetime
from typing import Dict, Any, Optional, List, Tuple

import structlog
from thefuzz import fuzz
from thefuzz import process

from core.utils.shared import TMP_DIR
from core.models.llm.structured import PageData

logger = structlog.get_logger(__name__)


class MappingSaver:
    """
    Handles the batch saving of Latin-to-Polish mappings for different fields.

    This class collects mappings, saves them to uniquely named JSON files in batches,
    and logs metadata to Weights & Biases.
    """
    _MAPPING_FILES = {
        "deanery": "deanery.json",
        "parish": "parish.json",
        "dedication": "dedication.json",
        "building_material": "building_material.json",
    }

    def __init__(self, batch_size: int = 5):
        """
        Initializes the MappingSaver.

        Args:
            batch_size (int): Number of pages to process before saving.
        """
        self.batch_size = batch_size
        
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = TMP_DIR / "mappings" / "generated" / f"run_{run_timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Mappings will be saved to: {self.save_dir}")

        self.pages_processed = 0
        self._mappings: Dict[str, Dict[str, str]] = {
            "deanery": {},
            "parish": {},
            "dedication": {},
            "building_material": {},
        }
        self._load_existing_mappings()
        self.idx = 0


    def _load_existing_mappings(self):
        """Loads existing mappings from the save directory if they exist."""
        for field, filename in self._MAPPING_FILES.items():
            filepath = self.save_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        self._mappings[field] = json.load(f)
                    logger.info(f"Loaded existing mappings for '{field}' from {filepath}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Could not load mappings for '{field}': {e}")

    def _match_entries(self, latin_entries: List[Any], polish_entries: List[Any]) -> List[Tuple[Any, Any]]:
        """
        Match Latin and Polish entries using fuzzy string matching on parish names.
        
        Args:
            latin_entries: List of Latin entries
            polish_entries: List of Polish entries
            
        Returns:
            List of matched (latin_entry, polish_entry) pairs
        """
        matched_pairs = []
        used_polish_indices = set()
        
        # For each Latin entry, find the best matching Polish entry based on parish name
        for latin_entry in latin_entries:
            if not latin_entry.parish:
                continue
                
            best_match = None
            best_score = 0
            best_idx = -1
            
            for i, polish_entry in enumerate(polish_entries):
                if i in used_polish_indices or not polish_entry.parish:
                    continue
                    
                score = fuzz.ratio(latin_entry.parish.lower(), polish_entry.parish.lower())
                if score > best_score and score > 80:  # 80% similarity threshold
                    best_score = score
                    best_match = polish_entry
                    best_idx = i
            
            if best_match:
                matched_pairs.append((latin_entry, best_match))
                used_polish_indices.add(best_idx)
        
        return matched_pairs

    def update(self, schematism: str, filename: str, latin_data: Dict[str, Any], polish_data: Dict[str, Any]):
        """
        Updates the internal mappings with new data from a page.

        Args:
            latin_data (dict): The dictionary with Latin values from the LLM.
            polish_data (dict): The dictionary with Polish ground truth values.
        """
        try:
            latin_page = PageData.parse_obj(latin_data)
            polish_page = PageData.parse_obj(polish_data)
        except Exception as e:
            logger.error("Failed to parse page data.", error=str(e), latin=latin_data, polish=polish_data)
            return

        if len(latin_page.entries) != len(polish_page.entries):
            logger.warning("Latin and Polish entries count mismatch.",
                           latin_count=len(latin_page.entries),
                           polish_count=len(polish_page.entries))
            
        # Match entries using fuzzy string matching
        matched_pairs = self._match_entries(latin_page.entries, polish_page.entries)
        
        # Process matched pairs
        for latin_entry, polish_entry in matched_pairs:
            entry_data = {
                'id': self.idx,
                'schematism': schematism,
                'filename': filename,
            }
            self.idx += 1
            
            for field in self._MAPPING_FILES.keys():
                latin_value = getattr(latin_entry, field)
                polish_value = getattr(polish_entry, field)

                if latin_value and polish_value:
                    self._mappings[field][latin_value] = polish_value
                    
                    # Save to CSV
                    csv_path = self.save_dir / f"{field}_mappings.csv"
                    csv_exists = csv_path.exists()
                    
                    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=['id', 'schematism', 'filename', 'latin', 'polish'])
                        if not csv_exists:
                            writer.writeheader()
                        writer.writerow({
                            **entry_data,
                            'latin': latin_value,
                            'polish': polish_value
                        })
        
        self.pages_processed += 1
        if self.pages_processed >= self.batch_size:
            self.save()

    def save(self, force: bool = False):
        """
        Saves the collected mappings to JSON files.

        Args:
            force (bool): If True, saves immediately regardless of batch size.
                          If False, saves only if batch size is reached.
        """
        if not force and self.pages_processed < self.batch_size:
            return

        if self.pages_processed == 0 and not force:
            return

        logger.info(f"Saving mappings to {self.save_dir}...")
        
        for field, mappings in self._mappings.items():
            filepath = self.save_dir / self._MAPPING_FILES[field]
            
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(mappings, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(mappings)} mappings for '{field}'")

            except (IOError, TypeError) as e:
                logger.error(f"Failed to save mappings for '{field}'.", error=str(e))
            
        self.pages_processed = 0 # Reset counter after saving
