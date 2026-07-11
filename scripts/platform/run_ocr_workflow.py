import argparse
import json
import tempfile
from pathlib import Path

from ocr_script_support import (
    PAGE_IMAGE_ARTIFACT_TYPE,
    PAGE_IMAGE_SCHEMA_VERSION,
    ApiClient,
    JsonObject,
    add_page_sequence_arguments,
    api_base_url_default,
    engine_config,
    object_field,
    object_list_field,
    resolve_page_sequence,
    validate_page_sequence_arguments,
)


def main() -> None:
    args = parse_args()
    client = ApiClient(args.api_base_url)
    with tempfile.TemporaryDirectory() as temp_dir:
        selection = resolve_page_sequence(client, args, Path(temp_dir))
        sequence = selection.sequence
        template_config: JsonObject = {
            "ocr": {
                "engine": args.engine,
                "language_hints": args.language_hint,
                "engine_config": engine_config(args.engine_config_json),
            }
        }
        if args.concrete_map:
            template_config["execution_planning"] = "concrete_map"

        launch = client.request_object(
            "POST",
            "/v1/workflow-templates/ocr-pages/launch",
            {
                "name": f"OCR pages with {args.engine}",
                "config": template_config,
                "input_artifact_sequence_refs": [
                    {
                        "sequence_id": sequence["id"],
                        "artifact_type": PAGE_IMAGE_ARTIFACT_TYPE,
                        "schema_version": PAGE_IMAGE_SCHEMA_VERSION,
                    }
                ],
                "metadata": {
                    "runner": "scripts/platform/run_ocr_workflow.py",
                    "source_sequence_id": str(sequence["id"]),
                },
                "change_note": "Created by scripts/platform/run_ocr_workflow.py",
            },
        )
        workflow = object_field(launch, "workflow_definition")
        version = object_field(launch, "workflow_version")
        run = object_field(launch, "workflow_run")

        max_node_runs = selection.source_page_count + 1 if args.concrete_map else 100
        execution = client.request_object(
            "POST",
            f"/v1/workflow-runs/{run['id']}/execute",
            {"max_node_runs": max_node_runs},
        )
        errors = execution["errors"]
        if errors != []:
            raise RuntimeError(f"Workflow run {run['id']} failed: {errors}")

        summary = client.request_object("GET", f"/v1/workflow-runs/{run['id']}/summary")
        outputs = client.request_object(
            "GET",
            f"/v1/workflow-runs/{run['id']}/outputs",
            query={"artifact_type": "ocr.page_result", "include_payloads": "true"},
        )
        ocr_payloads = _json_payloads(outputs)
        if len(ocr_payloads) != selection.source_page_count:
            raise RuntimeError(
                "OCR workflow produced "
                f"{len(ocr_payloads)} payloads for "
                f"{selection.source_page_count} source pages"
            )
        document_outputs = client.request_object(
            "GET",
            f"/v1/workflow-runs/{run['id']}/outputs",
            query={"artifact_type": "ocr.document_result", "include_payloads": "true"},
        )
        ocr_document_payloads = _json_payloads(document_outputs)
        if len(ocr_document_payloads) != 1:
            raise RuntimeError(
                "OCR workflow produced "
                f"{len(ocr_document_payloads)} document payloads"
            )

        print(
            json.dumps(
                {
                    "project_id": selection.project_id,
                    "source_id": selection.source_id,
                    "source_kind": selection.source_kind,
                    "source_page_count": selection.source_page_count,
                    "source_sequence_id": sequence["id"],
                    "workflow_id": workflow["id"],
                    "workflow_version_id": version["id"],
                    "workflow_run_id": run["id"],
                    "workflow_template_id": object_field(launch, "template")["id"],
                    "execution_planning": (
                        "concrete_map" if args.concrete_map else "sequence_node"
                    ),
                    "workflow_run_status": object_field(summary, "workflow_run")[
                        "status"
                    ],
                    "processed_node_run_ids": execution["processed_node_run_ids"],
                    "artifact_counts": summary["artifact_counts"],
                    "ocr_payloads": ocr_payloads,
                    "ocr_document_payload": ocr_document_payloads[0],
                },
                indent=2,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an OCR workflow through the Notarius HTTP API."
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    parser.add_argument(
        "--engine",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        default="local.tesseract",
    )
    add_page_sequence_arguments(parser)
    parser.add_argument("--language-hint", action="append", default=[])
    parser.add_argument(
        "--concrete-map",
        action="store_true",
        help="Run OCR as one node-run per page plus a collector node.",
    )
    parser.add_argument(
        "--engine-config-json",
        default="{}",
        help=(
            "JSON object passed to the OCR engine. For mistral.ocr this can include "
            "model, base_url, timeout_seconds, include_blocks, and api_key_env_var."
        ),
    )
    args = parser.parse_args()
    validate_page_sequence_arguments(parser, args)
    return args


def _json_payloads(outputs: JsonObject) -> list[JsonObject]:
    ocr_payloads: list[JsonObject] = []
    for output in object_list_field(outputs, "artifacts"):
        payload = output["payload"]
        if not isinstance(payload, dict):
            raise RuntimeError("Output artifact payload is not an object")
        payload_error = payload["error"]
        if payload_error is not None:
            raise RuntimeError(f"Output artifact payload failed: {payload_error}")
        ocr_payload = payload["json_payload"]
        if not isinstance(ocr_payload, dict):
            raise RuntimeError("Output artifact payload JSON is not an object")
        ocr_payloads.append(ocr_payload)
    return ocr_payloads


if __name__ == "__main__":
    main()
