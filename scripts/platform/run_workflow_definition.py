import argparse
import json
from pathlib import Path

from ocr_script_support import (
    ApiClient,
    JsonObject,
    api_base_url_default,
    json_object_config,
    object_field,
)


def main() -> None:
    args = parse_args()
    workflow_payload = read_json_object(args.workflow_json, "workflow_json")
    metadata = json_object_config(args.metadata_json, "--metadata-json")
    metadata["runner"] = "scripts/platform/run_workflow_definition.py"

    input_artifact_refs = [
        json_object_config(raw_ref, "--input-artifact-ref-json")
        for raw_ref in args.input_artifact_ref_json
    ]
    input_artifact_sequence_refs = [
        json_object_config(raw_ref, "--input-artifact-sequence-ref-json")
        for raw_ref in args.input_artifact_sequence_ref_json
    ]

    client = ApiClient(args.api_base_url)
    validation = client.request_object(
        "POST",
        "/v1/workflows/validate",
        workflow_payload,
    )
    if validation["valid"] is not True:
        raise RuntimeError(f"Workflow definition is invalid: {validation['errors']}")

    workflow = client.request_object("POST", "/v1/workflows", workflow_payload)
    version = client.request_object(
        "POST",
        f"/v1/workflows/{workflow['id']}/versions",
        {"change_note": args.change_note, "created_by": args.created_by},
    )
    run = client.request_object(
        "POST",
        "/v1/workflow-runs",
        {
            "workflow_version_id": version["id"],
            "input_artifact_refs": input_artifact_refs,
            "input_artifact_sequence_refs": input_artifact_sequence_refs,
            "metadata": metadata,
        },
    )
    execution = client.request_object(
        "POST",
        f"/v1/workflow-runs/{run['id']}/execute",
        {"max_node_runs": args.max_node_runs},
    )
    errors = execution["errors"]
    if errors != []:
        raise RuntimeError(f"Workflow run {run['id']} failed: {errors}")

    summary = client.request_object("GET", f"/v1/workflow-runs/{run['id']}/summary")
    output: JsonObject = {
        "workflow_id": workflow["id"],
        "workflow_version_id": version["id"],
        "workflow_run_id": run["id"],
        "workflow_run_status": object_field(summary, "workflow_run")["status"],
        "validation": validation,
        "processed_node_run_ids": execution["processed_node_run_ids"],
        "node_run_status_counts": summary["node_run_status_counts"],
        "artifact_counts": summary["artifact_counts"],
        "artifacts": summary["artifacts"],
    }

    if args.include_outputs:
        output_query: JsonObject = {
            "include_payloads": str(args.include_payloads).lower(),
            "include_text_payloads": str(args.include_text_payloads).lower(),
            "include_traces": str(args.include_traces).lower(),
        }
        if args.output_artifact_type is not None:
            output_query["artifact_type"] = args.output_artifact_type
        output["outputs"] = client.request_object(
            "GET",
            f"/v1/workflow-runs/{run['id']}/outputs",
            query=output_query,
        )

    print(json.dumps(output, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Notarius workflow definition JSON through the HTTP API."
    )
    parser.add_argument("workflow_json", type=Path)
    parser.add_argument("--api-base-url", default=api_base_url_default())
    parser.add_argument("--created-by")
    parser.add_argument(
        "--change-note",
        default="Created by scripts/platform/run_workflow_definition.py",
    )
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument(
        "--input-artifact-ref-json",
        action="append",
        default=[],
        help="JSON object matching ArtifactRefSchema. Repeat for multiple refs.",
    )
    parser.add_argument(
        "--input-artifact-sequence-ref-json",
        action="append",
        default=[],
        help="JSON object matching ArtifactSequenceRefSchema. Repeat for multiple refs.",
    )
    parser.add_argument("--max-node-runs", type=int, default=100)
    parser.add_argument("--include-outputs", action="store_true")
    parser.add_argument("--output-artifact-type")
    parser.add_argument("--include-payloads", action="store_true")
    parser.add_argument("--include-text-payloads", action="store_true")
    parser.add_argument("--include-traces", action="store_true")
    return parser.parse_args()


def read_json_object(path: Path, description: str) -> JsonObject:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{description} must decode to a JSON object")
    return decoded


if __name__ == "__main__":
    main()
