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
    json_object_config,
    object_field,
    object_list_field,
    resolve_page_sequence,
    validate_page_sequence_arguments,
)


DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "page_number": {"type": "integer"},
    },
    "required": ["text", "page_number"],
    "additionalProperties": False,
}
DEFAULT_PROMPT_TEMPLATE = (
    "Extract the requested structured record from page {{ CURRENT_PAGE_NUMBER }}.\n"
    "Current OCR text:\n{{ CURRENT_PAGE_TEXT }}\n"
    "Previous record: {{ PREVIOUS_RECORD }}"
)


def main() -> None:
    args = parse_args()
    client = ApiClient(args.api_base_url)
    with tempfile.TemporaryDirectory() as temp_dir:
        selection = resolve_page_sequence(client, args, Path(temp_dir))
        sequence = selection.sequence
        model_config: JsonObject = {
            "provider": args.model_provider,
            "model": args.model,
            "parameters": json_object_config(
                args.model_parameters_json,
                "--model-parameters-json",
            ),
            "capabilities": args.model_capability or ["structured_output"],
        }
        if args.credential_ref is not None:
            model_config["credential_ref"] = args.credential_ref
        if args.endpoint_ref is not None:
            model_config["endpoint_ref"] = args.endpoint_ref

        template_id = (
            "ocr-compare-contextual-extraction"
            if args.comparison_ocr_engine is not None
            else "contextual-extraction"
        )
        workflow_config: JsonObject = {
            "prompt": {
                "name": args.prompt_name,
                "template": args.prompt_template,
                "variables": [
                    "CURRENT_PAGE_NUMBER",
                    "CURRENT_PAGE_TEXT",
                    "PREVIOUS_RECORD",
                    "STATIC_CONTEXT",
                ],
            },
            "context": {
                "name": args.static_context_name,
                "context": json_object_config(
                    args.static_context_json,
                    "--static-context-json",
                ),
                "applies_to": ["text", "pages"],
            },
            "schema": {
                "name": args.schema_name,
                "json_schema": json_object_config(
                    args.schema_json,
                    "--schema-json",
                ),
            },
            "model": model_config,
            "policy": {
                "name": args.policy_name,
                "policy_type": args.policy_type,
                "settings": json_object_config(
                    args.policy_settings_json,
                    "--policy-settings-json",
                ),
                "applies_to": ["text", "pages"],
            },
            "extraction": {"result_key": args.result_key},
            "export": {"format": args.export_format},
        }
        if args.comparison_ocr_engine is None:
            workflow_config["ocr"] = {
                "engine": args.ocr_engine,
                "language_hints": args.language_hint,
                "engine_config": engine_config(args.engine_config_json),
            }
        else:
            workflow_config["ocr_a"] = {
                "engine": args.ocr_engine,
                "language_hints": args.language_hint,
                "engine_config": engine_config(args.engine_config_json),
            }
            workflow_config["ocr_b"] = {
                "engine": args.comparison_ocr_engine,
                "language_hints": args.language_hint,
                "engine_config": engine_config(args.comparison_engine_config_json),
            }
            workflow_config["compare"] = {
                "candidate_a_label": args.candidate_a_label,
                "candidate_b_label": args.candidate_b_label,
            }
            workflow_config["select"] = {
                "selected_candidate": args.selected_candidate,
            }
        launch = client.request_object(
            "POST",
            f"/v1/workflow-templates/{template_id}/launch",
            {
                "name": args.workflow_name,
                "config": workflow_config,
                "input_artifact_sequence_refs": [
                    {
                        "sequence_id": sequence["id"],
                        "artifact_type": PAGE_IMAGE_ARTIFACT_TYPE,
                        "schema_version": PAGE_IMAGE_SCHEMA_VERSION,
                    }
                ],
                "metadata": {
                    "runner": "scripts/platform/run_contextual_extraction_workflow.py",
                    "source_sequence_id": str(sequence["id"]),
                },
                "change_note": (
                    "Created by "
                    "scripts/platform/run_contextual_extraction_workflow.py"
                ),
            },
        )
        workflow = object_field(launch, "workflow_definition")
        version = object_field(launch, "workflow_version")
        run = object_field(launch, "workflow_run")
        execution = client.request_object(
            "POST",
            f"/v1/workflow-runs/{run['id']}/execute",
            {"max_node_runs": args.max_node_runs},
        )
        errors = execution["errors"]
        if errors != []:
            raise RuntimeError(f"Workflow run {run['id']} failed: {errors}")

        summary = client.request_object("GET", f"/v1/workflow-runs/{run['id']}/summary")
        outputs = client.request_object(
            "GET",
            f"/v1/workflow-runs/{run['id']}/outputs",
            query={"include_payloads": "true", "include_traces": "true"},
        )
        document_payloads = _json_payloads(outputs, "extraction.document_result")
        if len(document_payloads) != 1:
            raise RuntimeError(
                f"Expected one extraction.document_result, got {len(document_payloads)}"
            )
        document_payload = document_payloads[0]
        if document_payload["page_count"] != selection.source_page_count:
            raise RuntimeError(
                "Extraction document result page_count "
                f"{document_payload['page_count']} does not match source page count "
                f"{selection.source_page_count}"
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
                    "workflow_run_status": object_field(summary, "workflow_run")[
                        "status"
                    ],
                    "processed_node_run_ids": execution["processed_node_run_ids"],
                    "artifact_counts": summary["artifact_counts"],
                    "document_payload": document_payload,
                    "record_payloads": _json_payloads(
                        outputs,
                        "extraction.record_result",
                    ),
                    "static_context_payloads": _json_payloads(
                        outputs,
                        "context.bundle",
                    ),
                    "model_input_payloads": _json_payloads(outputs, "model.input"),
                    "model_response_payloads": _json_payloads(
                        outputs,
                        "model.response",
                    ),
                    "export_dataset_payloads": (
                        _json_payloads(outputs, "export.dataset")
                        if args.export_format == "json"
                        else []
                    ),
                    "export_dataset_texts": (
                        _text_payloads(outputs, "export.dataset")
                        if args.export_format in {"csv", "jsonl"}
                        else []
                    ),
                    "trace_counts": _trace_counts(outputs),
                },
                indent=2,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run OCR plus contextual structured extraction through the Notarius "
            "HTTP API."
        )
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    add_page_sequence_arguments(parser)
    parser.add_argument(
        "--ocr-engine",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        default="local.text",
    )
    parser.add_argument("--language-hint", action="append", default=[])
    parser.add_argument(
        "--engine-config-json",
        default="{}",
        help=(
            "JSON object passed to the OCR engine. For mistral.ocr this can include "
            "model, base_url, timeout_seconds, include_blocks, and api_key_env_var."
        ),
    )
    parser.add_argument(
        "--comparison-ocr-engine",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        help=(
            "Optional second OCR engine. When provided, the workflow compares "
            "both OCR streams, selects one, then extracts from the selected stream."
        ),
    )
    parser.add_argument(
        "--comparison-engine-config-json",
        default="{}",
        help="JSON object passed to --comparison-ocr-engine.",
    )
    parser.add_argument("--candidate-a-label", default="candidate_a")
    parser.add_argument("--candidate-b-label", default="candidate_b")
    parser.add_argument(
        "--selected-candidate",
        choices=("candidate_a", "candidate_b"),
        default="candidate_a",
    )
    parser.add_argument("--prompt-name", default="Script contextual extraction prompt")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--static-context-name", default="Script static context")
    parser.add_argument("--static-context-json", default="{}")
    parser.add_argument("--schema-name", default="Script contextual extraction schema")
    parser.add_argument("--schema-json", default=json.dumps(DEFAULT_SCHEMA))
    parser.add_argument("--model-provider", default="local")
    parser.add_argument("--model", default="echo")
    parser.add_argument("--model-capability", action="append", default=None)
    parser.add_argument("--model-parameters-json", default="{}")
    parser.add_argument("--credential-ref")
    parser.add_argument("--endpoint-ref")
    parser.add_argument("--policy-name", default="Script contextual extraction policy")
    parser.add_argument(
        "--policy-type",
        choices=("stateless", "accumulating", "sliding_window", "custom"),
        default="accumulating",
    )
    parser.add_argument("--policy-settings-json", default="{}")
    parser.add_argument("--result-key", default="record")
    parser.add_argument(
        "--export-format",
        choices=("json", "jsonl", "csv"),
        default="json",
    )
    parser.add_argument("--workflow-name", default="Script contextual extraction")
    parser.add_argument("--max-node-runs", type=int, default=100)
    args = parser.parse_args()
    validate_page_sequence_arguments(parser, args)
    return args


def _json_payloads(outputs: JsonObject, artifact_type: str) -> list[JsonObject]:
    payloads: list[JsonObject] = []
    for output in object_list_field(outputs, "artifacts"):
        artifact = object_field(output, "artifact")
        if artifact["artifact_type"] != artifact_type:
            continue
        payload = output["payload"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"{artifact_type} output artifact payload is not an object")
        payload_error = payload["error"]
        if payload_error is not None:
            raise RuntimeError(f"{artifact_type} output payload failed: {payload_error}")
        json_payload = payload["json_payload"]
        if not isinstance(json_payload, dict):
            raise RuntimeError(f"{artifact_type} output JSON payload is not an object")
        payloads.append(json_payload)
    return payloads


def _text_payloads(outputs: JsonObject, artifact_type: str) -> list[str]:
    payloads: list[str] = []
    for output in object_list_field(outputs, "artifacts"):
        artifact = object_field(output, "artifact")
        if artifact["artifact_type"] != artifact_type:
            continue
        payload = output["payload"]
        if not isinstance(payload, dict):
            raise RuntimeError(f"{artifact_type} output artifact payload is not an object")
        payload_error = payload["error"]
        if payload_error is not None:
            raise RuntimeError(f"{artifact_type} output payload failed: {payload_error}")
        text_payload = payload["text"]
        if not isinstance(text_payload, str):
            raise RuntimeError(f"{artifact_type} output text payload is not a string")
        payloads.append(text_payload)
    return payloads


def _trace_counts(outputs: JsonObject) -> JsonObject:
    input_assembly_trace_count = 0
    invocation_trace_count = 0
    for trace_bundle in object_list_field(outputs, "traces"):
        input_assembly_trace_count += len(
            object_list_field(trace_bundle, "input_assembly_traces")
        )
        invocation_trace_count += len(object_list_field(trace_bundle, "invocation_traces"))
    return {
        "input_assembly": input_assembly_trace_count,
        "invocation": invocation_trace_count,
    }


if __name__ == "__main__":
    main()
