import argparse
import json
from pathlib import Path

from ocr_script_support import ApiClient, api_base_url_default


def main() -> None:
    args = parse_args()
    decoded = json.loads(args.workflow_json.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Workflow JSON must decode to an object")

    client = ApiClient(args.api_base_url)
    validation = client.request_object(
        "POST",
        "/v1/workflows/validate",
        decoded,
    )
    print(json.dumps(validation, indent=2))

    if args.fail_on_invalid and validation["valid"] is not True:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Notarius workflow definition through the HTTP API."
    )
    parser.add_argument("workflow_json", type=Path)
    parser.add_argument("--api-base-url", default=api_base_url_default())
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit with status 1 when the API returns valid=false.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
