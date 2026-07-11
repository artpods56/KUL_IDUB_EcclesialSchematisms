import json
import shutil
from pathlib import Path
from typing import override

from dagster import ConfigurableIOManager, InputContext, OutputContext
from datasets import load_from_disk, Dataset, DatasetDict
from notarius.shared.constants import TMP_DIR


class HuggingFaceDatasetIOManager(ConfigurableIOManager):
    """IO Manager for Hugging Face Datasets using save_to_disk and load_from_disk.

    This manager avoids pickling HF Dataset objects, which can lead to
    memory-mapping issues and race conditions in shared environments.
    """

    base_dir: str = str(TMP_DIR / "dagster_hf_storage")

    def _get_path(self, context: OutputContext | InputContext) -> Path:
        """Generate directory path based on asset key."""
        parts = context.asset_key.path if context.asset_key else ["output"]
        return Path(self.base_dir, *parts)

    @override
    def handle_output(self, context: OutputContext, obj: Dataset) -> None:
        """Store the Dataset using save_to_disk."""
        if obj is None:
            return

        filepath = self._get_path(context)
        context.log.info(f"Saving Hugging Face Dataset to: {filepath}")

        # Handle empty datasets separately due to HuggingFace bug with save_to_disk on empty datasets
        if len(obj) == 0:
            context.log.warning("Dataset is empty (0 rows). Saving empty dataset marker.")
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing directory if it exists
            if filepath.exists():
                shutil.rmtree(filepath)

            filepath.mkdir(parents=True, exist_ok=True)

            # Save metadata about the empty dataset (schema, features, etc.)
            empty_marker = filepath / ".empty_dataset"
            metadata = {
                "num_rows": 0,
                "features": (
                    {name: str(feature) for name, feature in obj.features.items()}
                    if obj.features
                    else {}
                ),
                "empty": True,
            }
            empty_marker.write_text(json.dumps(metadata, indent=2))
            context.log.info(f"Saved empty dataset marker to: {empty_marker}")
            return

        # Save to a temporary directory first to ensure atomicity
        # Use a unique temporary name to avoid collisions between runs/assets
        tmp_path = filepath.parent / f".tmp_{filepath.name}_{context.run_id}"

        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        tmp_path.mkdir(parents=True, exist_ok=True)

        try:
            obj.save_to_disk(str(tmp_path))

            # Atomic swap: remove old directory and move new one
            if filepath.exists():
                context.log.debug(f"Removing existing directory: {filepath}")
                shutil.rmtree(filepath)

            filepath.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_path), str(filepath))
            context.log.info(f"Successfully saved and moved dataset to: {filepath}")

        except Exception as e:
            context.log.error(f"Failed to save dataset: {e}")
            if tmp_path.exists():
                try:
                    shutil.rmtree(tmp_path)
                except Exception:
                    pass
            raise

    @override
    def load_input(self, context: InputContext) -> Dataset | DatasetDict:
        """Load the Dataset using load_from_disk."""
        filepath = self._get_path(context)
        context.log.info(f"Loading Hugging Face Dataset from: {filepath}")

        if not filepath.exists():
            raise FileNotFoundError(f"Hugging Face Dataset not found at {filepath}")

        # Check if this is an empty dataset marker
        empty_marker = filepath / ".empty_dataset"
        if empty_marker.exists():
            context.log.warning("Loading empty dataset from marker file")
            metadata = json.loads(empty_marker.read_text())
            context.log.info(f"Empty dataset metadata: {metadata}")

            # Create an empty dataset with the preserved schema if available
            # For now, return a minimal empty dataset
            return Dataset.from_dict({})

        return load_from_disk(str(filepath))


def hf_dataset_io_manager(base_dir: str | None = None) -> HuggingFaceDatasetIOManager:
    """Factory function to create a HuggingFaceDatasetIOManager."""
    if base_dir is None:
        base_dir = str(TMP_DIR / "dagster_hf_storage")
    return HuggingFaceDatasetIOManager(base_dir=base_dir)
