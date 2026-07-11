import argparse
import csv
import json
from io import StringIO
from pathlib import Path

from ocr_script_support import (
    ApiClient,
    JsonObject,
    api_base_url_default,
    object_field,
)


def main() -> None:
    args = parse_args()
    client = ApiClient(args.api_base_url)

    if args.command == "get":
        experiment = client.request_object(
            "GET",
            f"/v1/experiments/{args.experiment_id}",
        )
        print(json.dumps({"experiment": experiment}, indent=2))
        return

    if args.command == "comparison":
        comparison = client.request_object(
            "GET",
            f"/v1/experiments/{args.experiment_id}/comparison",
        )
        if args.output_format == "json":
            output = json.dumps({"comparison": comparison}, indent=2) + "\n"
        else:
            output = comparison_csv(comparison)
        write_or_print(output, args.output_path)
        return

    if args.command == "outputs":
        query: JsonObject = {
            "include_payloads": str(args.include_payloads).lower(),
            "include_text_payloads": str(args.include_text_payloads).lower(),
            "include_traces": str(args.include_traces).lower(),
        }
        if args.artifact_type is not None:
            query["artifact_type"] = args.artifact_type
        outputs = client.request_object(
            "GET",
            f"/v1/experiments/{args.experiment_id}/outputs",
            query=query,
        )
        output = json.dumps({"outputs": outputs}, indent=2) + "\n"
        write_or_print(output, args.output_path)
        return

    if args.command == "events":
        experiment = client.request_object(
            "GET",
            f"/v1/experiments/{args.experiment_id}",
        )
        experiment_events = {
            "experiment": experiment,
            "variant_timelines": experiment_variant_timelines(client, experiment),
        }
        output = json.dumps({"experiment_events": experiment_events}, indent=2) + "\n"
        write_or_print(output, args.output_path)
        return

    if args.command == "cancel":
        experiment = client.request_object(
            "POST",
            f"/v1/experiments/{args.experiment_id}/cancel",
        )
        print(json.dumps({"experiment": experiment}, indent=2))
        return

    if args.command == "rerun-failed":
        rerun = client.request_object(
            "POST",
            f"/v1/experiments/{args.experiment_id}/rerun-failed",
        )
        print(json.dumps({"rerun": rerun}, indent=2))
        return

    if args.command == "cancel-variant":
        experiment = client.request_object(
            "POST",
            (
                f"/v1/experiments/{args.experiment_id}/variants/"
                f"{args.variant_id}/cancel"
            ),
        )
        print(json.dumps({"experiment": experiment}, indent=2))
        return

    rerun = client.request_object(
        "POST",
        (
            f"/v1/experiments/{args.experiment_id}/variants/"
            f"{args.variant_id}/rerun"
        ),
    )
    print(json.dumps({"experiment": rerun}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and operate Notarius experiments through the HTTP API."
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    subparsers = parser.add_subparsers(dest="command", required=True)

    get_parser = subparsers.add_parser("get", help="Read one experiment.")
    get_parser.add_argument("experiment_id")

    comparison_parser = subparsers.add_parser(
        "comparison",
        help="Read the experiment comparison table.",
    )
    comparison_parser.add_argument("experiment_id")
    comparison_parser.add_argument(
        "--output-format",
        choices=("json", "csv"),
        default="json",
    )
    comparison_parser.add_argument(
        "--output-path",
        type=Path,
        help="Write the comparison output to a file instead of stdout.",
    )

    outputs_parser = subparsers.add_parser(
        "outputs",
        help="Read variant output bundles for an experiment.",
    )
    outputs_parser.add_argument("experiment_id")
    outputs_parser.add_argument("--artifact-type")
    outputs_parser.add_argument("--include-payloads", action="store_true")
    outputs_parser.add_argument("--include-text-payloads", action="store_true")
    outputs_parser.add_argument("--include-traces", action="store_true")
    outputs_parser.add_argument(
        "--output-path",
        type=Path,
        help="Write the output bundle JSON to a file instead of stdout.",
    )

    events_parser = subparsers.add_parser(
        "events",
        help="Read current workflow-run event timelines for every experiment variant.",
    )
    events_parser.add_argument("experiment_id")
    events_parser.add_argument(
        "--output-path",
        type=Path,
        help="Write the event timeline JSON to a file instead of stdout.",
    )

    cancel_parser = subparsers.add_parser(
        "cancel",
        help="Cancel all open variant workflow runs for an experiment.",
    )
    cancel_parser.add_argument("experiment_id")

    rerun_failed_parser = subparsers.add_parser(
        "rerun-failed",
        help="Rerun failed experiment variants without mutating prior artifacts.",
    )
    rerun_failed_parser.add_argument("experiment_id")

    cancel_variant_parser = subparsers.add_parser(
        "cancel-variant",
        help="Cancel one experiment variant's current workflow run.",
    )
    cancel_variant_parser.add_argument("experiment_id")
    cancel_variant_parser.add_argument("variant_id")

    rerun_variant_parser = subparsers.add_parser(
        "rerun-variant",
        help="Rerun one failed or cancelled experiment variant.",
    )
    rerun_variant_parser.add_argument("experiment_id")
    rerun_variant_parser.add_argument("variant_id")
    return parser.parse_args()


def experiment_variant_timelines(
    client: ApiClient,
    experiment: JsonObject,
) -> list[JsonObject]:
    timelines: list[JsonObject] = []
    for variant in _object_list(experiment, "variants"):
        workflow_run_id = variant["workflow_run_id"]
        timeline = client.request_object(
            "GET",
            f"/v1/workflow-runs/{workflow_run_id}/events",
        )
        timelines.append(
            {
                "variant_id": variant["id"],
                "variant_key": variant["key"],
                "ordinal": variant["ordinal"],
                "parameter_values": object_field(variant, "parameter_values"),
                "workflow_run_id": workflow_run_id,
                "timeline": timeline,
            }
        )
    return timelines


def comparison_csv(comparison: JsonObject) -> str:
    rows = comparison_rows(comparison)
    columns = comparison_columns(comparison, rows)
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def comparison_rows(comparison: JsonObject) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for variant in _object_list(comparison, "variants"):
        row: JsonObject = {
            "variant_key": variant["variant_key"],
            "workflow_run_id": variant["workflow_run_id"],
            "workflow_run_status": variant["workflow_run_status"],
            "validation_error_count": variant["validation_error_count"],
            "total_duration_ms": variant["total_duration_ms"],
            "total_cost": variant["total_cost"],
        }
        for name, value in _prefixed_fields(
            object_field(variant, "parameter_values"),
            "param.",
        ).items():
            row[name] = value
        for name, value in _prefixed_fields(
            object_field(variant, "artifact_counts"),
            "artifact_count.",
        ).items():
            row[name] = value
        for metric in _object_list(variant, "metric_values"):
            metric_name = metric["name"]
            if not isinstance(metric_name, str):
                raise RuntimeError("metric_values contains a non-string name")
            row[f"metric.{metric_name}"] = metric["value"]
        rows.append(row)
    return rows


def comparison_columns(comparison: JsonObject, rows: list[JsonObject]) -> list[str]:
    columns = [
        "variant_key",
        "workflow_run_id",
        "workflow_run_status",
        "validation_error_count",
        "total_duration_ms",
        "total_cost",
    ]
    for row in rows:
        for key in row:
            if key.startswith("param.") and key not in columns:
                columns.append(key)
    for row in rows:
        for key in row:
            if key.startswith("artifact_count.") and key not in columns:
                columns.append(key)
    for metric_name in _string_list(comparison, "metric_names"):
        metric_column = f"metric.{metric_name}"
        if metric_column not in columns:
            columns.append(metric_column)
    return columns


def write_or_print(output: str, output_path: Path | None) -> None:
    if output_path is None:
        print(output, end="")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output, encoding="utf-8")
    print(json.dumps({"output_path": str(output_path)}, indent=2))


def _prefixed_fields(value: JsonObject, prefix: str) -> JsonObject:
    return {
        f"{prefix}{key}": field_value
        for key, field_value in sorted(value.items())
    }


def _object_list(value: JsonObject, field_name: str) -> list[JsonObject]:
    field_value = value[field_name]
    if not isinstance(field_value, list):
        raise RuntimeError(f"{field_name} is not a list")
    result = []
    for item in field_value:
        if not isinstance(item, dict):
            raise RuntimeError(f"{field_name} contains a non-object item")
        result.append(item)
    return result


def _string_list(value: JsonObject, field_name: str) -> list[str]:
    field_value = value[field_name]
    if not isinstance(field_value, list):
        raise RuntimeError(f"{field_name} is not a list")
    result = []
    for item in field_value:
        if not isinstance(item, str):
            raise RuntimeError(f"{field_name} contains a non-string item")
        result.append(item)
    return result


if __name__ == "__main__":
    main()
