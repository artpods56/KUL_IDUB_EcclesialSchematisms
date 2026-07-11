import argparse
import json
from pathlib import Path

from notarius_api.main import app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the isolated Notarius prototype OpenAPI schema.",
    )
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            app.openapi(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
