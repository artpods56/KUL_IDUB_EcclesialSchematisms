import argparse
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


JsonObject = dict[str, object]


def request_json(
    method: str,
    path: str,
    payload: JsonObject | None = None,
) -> object:
    base_url = os.getenv("NOTARIUS_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        f"{base_url}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=120) as response:
            decoded = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise RuntimeError(f"{method} {path} failed: {exc.code} {detail}") from exc

    return json.loads(decoded)


def request_object(
    method: str,
    path: str,
    payload: JsonObject | None = None,
) -> JsonObject:
    parsed = request_json(method, path, payload)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {path} returned a non-object JSON response")
    return parsed


def main() -> None:
    args = parse_args()
    workflow = request_object(
        "POST",
        "/v1/workflows",
        {
            "name": args.workflow_name,
            "nodes": [
                {
                    "id": "emit",
                    "operator_id": "debug.emit_text",
                    "operator_version": "1.0.0",
                    "config": {"text": "baseline text"},
                }
            ],
        },
    )
    version = request_object(
        "POST",
        f"/v1/workflows/{workflow['id']}/versions",
        {"change_note": "Created by scripts/platform/run_debug_experiment.py"},
    )
    experiment = request_object(
        "POST",
        "/v1/experiments",
        {
            "name": args.experiment_name,
            "workflow_version_id": version["id"],
            "parameters": [
                {
                    "name": "text",
                    "node_id": "emit",
                    "config_path": ["text"],
                    "values": args.value,
                }
            ],
            "metadata": {"runner": "scripts/platform/run_debug_experiment.py"},
        },
    )

    execution = request_object(
        "POST",
        f"/v1/experiments/{experiment['id']}/execute",
        {
            "max_node_runs_per_variant": args.max_node_runs_per_variant,
            "stop_on_error": args.stop_on_error,
        },
    )
    failed_variants = [
        variant
        for variant in _object_list_field(execution, "variants")
        if variant["errors"] != []
    ]
    if failed_variants:
        raise RuntimeError(
            f"Experiment {experiment['id']} failed variants: {failed_variants}"
        )

    comparison = request_object(
        "GET",
        f"/v1/experiments/{experiment['id']}/comparison",
    )
    outputs = request_object(
        "GET",
        f"/v1/experiments/{experiment['id']}/outputs?include_traces=true",
    )
    print(
        json.dumps(
            {
                "workflow_id": workflow["id"],
                "workflow_version_id": version["id"],
                "experiment_id": experiment["id"],
                "experiment_status": _object_field(execution, "experiment")["status"],
                "variant_count": comparison["variant_count"],
                "metric_names": comparison["metric_names"],
                "variants": [
                    {
                        "variant_key": variant["variant_key"],
                        "parameter_values": variant["parameter_values"],
                        "workflow_run_id": variant["workflow_run_id"],
                        "workflow_run_status": variant["workflow_run_status"],
                        "processed_node_run_ids": _processed_node_run_ids(
                            execution,
                            variant["variant_id"],
                        ),
                        "artifact_counts": variant["artifact_counts"],
                        "metric_values": variant["metric_values"],
                        "output_artifacts": _output_artifacts(
                            outputs,
                            variant["variant_id"],
                        ),
                    }
                    for variant in _object_list_field(comparison, "variants")
                ],
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a debug experiment matrix through the Notarius HTTP API."
    )
    parser.add_argument("--workflow-name", default="Script debug experiment workflow")
    parser.add_argument("--experiment-name", default="Script debug experiment")
    parser.add_argument(
        "--value",
        action="append",
        default=None,
        help="Variant text value. Repeat to create multiple variants.",
    )
    parser.add_argument("--max-node-runs-per-variant", type=int, default=100)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    if args.value is None:
        args.value = ["variant A from script", "variant B from script"]
    return args


def _object_field(value: object, field_name: str) -> JsonObject:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object while reading field {field_name}")
    field_value = value[field_name]
    if not isinstance(field_value, dict):
        raise RuntimeError(f"Field {field_name} is not an object")
    return field_value


def _object_list_field(value: object, field_name: str) -> list[JsonObject]:
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected object while reading field {field_name}")
    field_value = value[field_name]
    if not isinstance(field_value, list):
        raise RuntimeError(f"Field {field_name} is not a list")
    objects: list[JsonObject] = []
    for item in field_value:
        if not isinstance(item, dict):
            raise RuntimeError(f"Field {field_name} contains a non-object item")
        objects.append(item)
    return objects


def _processed_node_run_ids(
    execution: JsonObject,
    variant_id: object,
) -> list[object]:
    for variant in _object_list_field(execution, "variants"):
        if variant["variant_id"] == variant_id:
            processed_ids = variant["processed_node_run_ids"]
            if not isinstance(processed_ids, list):
                raise RuntimeError("processed_node_run_ids is not a list")
            return processed_ids
    raise RuntimeError(f"Variant {variant_id} missing from execution response")


def _output_artifacts(
    outputs: JsonObject,
    variant_id: object,
) -> list[JsonObject]:
    for variant in _object_list_field(outputs, "variants"):
        if variant["variant_id"] != variant_id:
            continue
        output_bundle = variant["output_bundle"]
        if not isinstance(output_bundle, dict):
            raise RuntimeError("output_bundle is not an object")
        return _object_list_field(output_bundle, "artifacts")
    raise RuntimeError(f"Variant {variant_id} missing from outputs response")


if __name__ == "__main__":
    main()
