#!/usr/bin/env python3
"""
Script to convert old source JSON structure to new format.

Old structure:
{
  "records": [
    {
      "sample_id": 123,
      "filename": "0001.jpg",
      "schematism_name": "wloclawek_1872",
      "source": { ... }
    }
  ]
}

New structure:
{
  "records": [
    {
      "predictions": { ... },
      "image_path": "",
      "text": "",
      "metadata": {
        "sample_id": 123,
        "schematism_name": "wloclawek_1872",
        "filename": "0001.jpg"
      }
    }
  ]
}

Usage:
    python scripts/convert_source_json_structure.py <input_file> [output_file]

    If output_file is not specified, it will add '_converted' suffix to input filename.
"""

import json
import sys
from pathlib import Path
from typing import Any


def convert_record(old_record: dict[str, Any]) -> dict[str, Any]:
    """Convert a single record from old to new structure."""
    new_record = {
        "predictions": old_record["source"],
        "image_path": "",  # Not available in old format
        "text": "",  # Not available in old format
        "metadata": {
            "sample_id": old_record["sample_id"],
            "schematism_name": old_record["schematism_name"],
            "filename": old_record["filename"],
        },
    }
    return new_record


def convert_json_file(input_path: Path, output_path: Path) -> None:
    """Convert entire JSON file from old to new structure."""
    print(f"Reading from: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert all records
    converted_records = [convert_record(record) for record in data["records"]]

    # Build new structure
    new_data = {
        "generated_at": data["generated_at"],
        "total_records": data["total_records"],
        "records": converted_records,
    }

    print(f"Converting {len(converted_records)} records...")

    # Write to output file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print(f"Written to: {output_path}")
    print(f"Total records converted: {len(converted_records)}")


def main():
    if len(sys.argv) < 2:
        print("Error: Input file required")
        print(f"Usage: {sys.argv[0]} <input_file> [output_file]")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        # Add '_converted' suffix before extension
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = input_path.parent / f"{stem}_converted{suffix}"

    # Check if output file already exists
    if output_path.exists():
        response = input(f"Output file {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    try:
        convert_json_file(input_path, output_path)
        print("\nConversion complete!")
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
