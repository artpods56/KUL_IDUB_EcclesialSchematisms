import argparse
import json
from pathlib import Path

from fastapi import FastAPI

from notarius_api.v1.routes.prototype import router as prototype_router


prototype_openapi_app = FastAPI(
    title="Notarius Prototype API",
    version="0.1.0",
)
prototype_openapi_app.include_router(prototype_router, prefix="/v1")


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
            prototype_openapi_app.openapi(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
