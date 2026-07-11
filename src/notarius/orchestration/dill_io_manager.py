"""Dagster IO Manager using dill for serialization.

Dill extends pickle to serialize a wider range of Python objects including:
- Lambda functions
- Nested functions
- Class instances with complex state
- Generic types (e.g., BaseDataset[GroundTruthDataItem])

This is particularly useful for ML/AI pipelines where objects may contain
complex state, custom classes, or functions that pickle cannot handle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import dill
from dagster import ConfigurableIOManager, InputContext, OutputContext
from notarius.shared.constants import TMP_DIR


class DillIOManager(ConfigurableIOManager):
    """IO Manager that uses dill for serialization instead of pickle."""

    base_dir: str = str(TMP_DIR / "dagster_io")

    def _get_path(self, context: OutputContext | InputContext) -> Path:
        """Generate file path based on asset key."""
        parts = context.asset_key.path if context.asset_key else ["output"]
        directory = Path(self.base_dir, *parts[:-1])
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{parts[-1]}.dill"

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Store the output object using dill serialization."""
        if obj is None:
            return

        filepath = self._get_path(context)
        context.log.info(f"Writing dill file: {filepath}")

        with open(filepath, "wb") as f:
            dill.dump(obj, f)

    def load_input(self, context: InputContext) -> Any:
        """Load the input object using dill deserialization."""
        filepath = self._get_path(context)
        context.log.info(f"Loading dill file: {filepath}")

        if not filepath.exists():
            raise FileNotFoundError(f"Asset file not found: {filepath}")

        with open(filepath, "rb") as f:
            return dill.load(f)


def dill_io_manager(base_dir: str | None = None) -> DillIOManager:
    """Factory function to create a DillIOManager."""
    if base_dir is None:
        base_dir = str(TMP_DIR / "dagster_io")
    return DillIOManager(base_dir=base_dir)
