"""Utilities for handling and saving generated mappings."""

import json
import datetime
from typing import Dict, Any, Optional

import structlog
import wandb

from core.utils.shared import TMP_DIR
from core.schemas.data.schematism import SchematismPage

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

    def __init__(self, batch_size: int = 5, wandb_run: Optional[wandb.sdk.wandb_run.Run] = None):
        """
        Initializes the MappingSaver.

        Args:
            batch_size (int): Number of pages to process before saving.
            wandb_run (wandb.Run, optional): Active W&B run for logging.
        """
        self.batch_size = batch_size
        self.wandb_run = wandb_run
        
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

    def update(self, latin_data: Dict[str, Any], polish_data: Dict[str, Any]):
        """
        Updates the internal mappings with new data from a page.

        Args:
            latin_data (dict): The dictionary with Latin values from the LLM.
            polish_data (dict): The dictionary with Polish ground truth values.
        """
        try:
            latin_page = SchematismPage.parse_obj(latin_data)
            polish_page = SchematismPage.parse_obj(polish_data)
        except Exception as e:
            logger.error("Failed to parse page data.", error=str(e), latin=latin_data, polish=polish_data)
            return

        for latin_entry, polish_entry in zip(latin_page.entries, polish_page.entries):
            for field in self._MAPPING_FILES.keys():
                latin_value = getattr(latin_entry, field)
                polish_value = getattr(polish_entry, field)

                if latin_value and polish_value:
                    self._mappings[field][latin_value] = polish_value
        
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
        
        wandb_logs = {}
        for field, mappings in self._mappings.items():
            filepath = self.save_dir / self._MAPPING_FILES[field]
            
            try:
                # To calculate new mappings, we load the file again and compare sizes
                # This is safer in case of concurrent writes or multiple instances
                current_count = 0
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        current_count = len(json.load(f))
                
                new_count = len(mappings) - current_count
                if new_count > 0:
                     wandb_logs[f"mappings/new_{field}"] = new_count
                
                wandb_logs[f"mappings/total_{field}"] = len(mappings)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(mappings, f, ensure_ascii=False, indent=2)

            except (IOError, TypeError) as e:
                logger.error(f"Failed to save mappings for '{field}'.", error=str(e))

        if self.wandb_run and wandb_logs:
            self.wandb_run.log(wandb_logs)
            logger.info("Logged mapping counts to W&B.", **wandb_logs)
            
        self.pages_processed = 0 # Reset counter after saving
