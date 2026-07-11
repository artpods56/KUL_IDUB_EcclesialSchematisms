import argparse
import json
from pathlib import Path

from ocr_script_support import ApiClient, JsonObject, api_base_url_default


def main() -> None:
    args = parse_args()
    client = ApiClient(args.api_base_url)

    if args.command == "get":
        artifact = client.request_object("GET", f"/v1/artifacts/{args.artifact_id}")
        output = json.dumps({"artifact": artifact}, indent=2) + "\n"
        write_or_print(output, args.output_path)
        return

    if args.command == "lineage":
        lineage = client.request_object(
            "GET",
            f"/v1/artifacts/{args.artifact_id}/lineage",
        )
        output = json.dumps({"lineage": lineage}, indent=2) + "\n"
        write_or_print(output, args.output_path)
        return

    query: JsonObject = {
        "include_payload": str(args.include_payload).lower(),
        "include_text_payload": str(args.include_text_payload).lower(),
        "include_lineage": str(args.include_lineage).lower(),
    }
    inspection = client.request_object(
        "GET",
        f"/v1/artifacts/{args.artifact_id}/inspect",
        query=query,
    )
    output = json.dumps({"inspection": inspection}, indent=2) + "\n"
    write_or_print(output, args.output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Notarius artifacts through the HTTP API."
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="Read artifact metadata.")
    get_parser.add_argument("artifact_id")
    get_parser.add_argument(
        "--output-path",
        type=Path,
        help="Write the artifact JSON to a file instead of stdout.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Read artifact metadata with optional decoded payload and lineage.",
    )
    inspect_parser.add_argument("artifact_id")
    inspect_parser.add_argument("--include-payload", action="store_true")
    inspect_parser.add_argument("--include-text-payload", action="store_true")
    inspect_parser.add_argument("--include-lineage", action="store_true")
    inspect_parser.add_argument(
        "--output-path",
        type=Path,
        help="Write the inspection JSON to a file instead of stdout.",
    )

    lineage_parser = subparsers.add_parser(
        "lineage",
        help="Read the artifact lineage graph.",
    )
    lineage_parser.add_argument("artifact_id")
    lineage_parser.add_argument(
        "--output-path",
        type=Path,
        help="Write the lineage JSON to a file instead of stdout.",
    )
    return parser.parse_args()


def write_or_print(output: str, output_path: Path | None) -> None:
    if output_path is None:
        print(output, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(json.dumps({"output_path": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
