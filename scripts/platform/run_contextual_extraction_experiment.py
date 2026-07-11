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
        template_id = _workflow_template_id(args)
        materialized = client.request_object(
            "POST",
            f"/v1/workflow-templates/{template_id}/materialize",
            {
                "name": args.workflow_name,
                "config": _workflow_config(args),
                "metadata": {
                    "runner": (
                        "scripts/platform/"
                        "run_contextual_extraction_experiment.py"
                    ),
                    "source_sequence_id": str(sequence["id"]),
                },
                "change_note": (
                    "Created by scripts/platform/"
                    "run_contextual_extraction_experiment.py"
                ),
            },
        )
        workflow = object_field(materialized, "workflow_definition")
        version = object_field(materialized, "workflow_version")
        experiment = client.request_object(
            "POST",
            "/v1/experiments",
            {
                "name": args.experiment_name,
                "workflow_version_id": version["id"],
                "parameter_presets": _parameter_presets(args),
                "input_artifact_sequence_refs": [
                    {
                        "sequence_id": sequence["id"],
                        "artifact_type": PAGE_IMAGE_ARTIFACT_TYPE,
                        "schema_version": PAGE_IMAGE_SCHEMA_VERSION,
                    }
                ],
                "metadata": {
                    "runner": (
                        "scripts/platform/"
                        "run_contextual_extraction_experiment.py"
                    ),
                    "source_id": selection.source_id,
                    "source_sequence_id": sequence["id"],
                    "source_page_count": selection.source_page_count,
                    "workflow_template_id": object_field(materialized, "template")[
                        "id"
                    ],
                },
            },
        )
        execution = client.request_object(
            "POST",
            f"/v1/experiments/{experiment['id']}/execute",
            {
                "max_node_runs_per_variant": args.max_node_runs_per_variant,
                "stop_on_error": args.stop_on_error,
            },
        )
        failed_variants = [
            variant
            for variant in object_list_field(execution, "variants")
            if variant["errors"] != []
        ]
        if failed_variants:
            raise RuntimeError(
                f"Experiment {experiment['id']} failed variants: {failed_variants}"
            )

        comparison = client.request_object(
            "GET",
            f"/v1/experiments/{experiment['id']}/comparison",
        )
        outputs = client.request_object(
            "GET",
            f"/v1/experiments/{experiment['id']}/outputs",
            query={"include_payloads": "true", "include_traces": "true"},
        )
        variants = [
            _variant_summary(
                variant,
                execution,
                outputs,
                expected_page_count=selection.source_page_count,
            )
            for variant in object_list_field(comparison, "variants")
        ]
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
                    "workflow_template_id": object_field(materialized, "template")[
                        "id"
                    ],
                    "experiment_id": experiment["id"],
                    "experiment_status": object_field(execution, "experiment")[
                        "status"
                    ],
                    "variant_count": comparison["variant_count"],
                    "metric_names": comparison["metric_names"],
                    "variants": variants,
                },
                indent=2,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run contextual structured extraction experiment variants through "
            "the Notarius HTTP API."
        )
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    add_page_sequence_arguments(parser)
    parser.add_argument(
        "--ocr-engine",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        default="local.text",
    )
    parser.add_argument(
        "--ocr-engine-variant",
        action="append",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        default=[],
    )
    parser.add_argument("--language-hint", action="append", default=[])
    parser.add_argument("--engine-config-json", default="{}")
    parser.add_argument("--engine-config-json-variant", action="append", default=[])
    parser.add_argument(
        "--comparison-ocr-engine",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        help=(
            "Optional second OCR engine. When provided, the experiment uses the "
            "OCR-compare contextual extraction template."
        ),
    )
    parser.add_argument(
        "--comparison-ocr-engine-variant",
        action="append",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        default=[],
    )
    parser.add_argument("--comparison-engine-config-json", default="{}")
    parser.add_argument(
        "--comparison-engine-config-json-variant",
        action="append",
        default=[],
    )
    parser.add_argument("--candidate-a-label", default="candidate_a")
    parser.add_argument("--candidate-a-label-variant", action="append", default=[])
    parser.add_argument("--candidate-b-label", default="candidate_b")
    parser.add_argument("--candidate-b-label-variant", action="append", default=[])
    parser.add_argument(
        "--selected-candidate",
        choices=("candidate_a", "candidate_b"),
        default="candidate_a",
    )
    parser.add_argument(
        "--selected-candidate-variant",
        action="append",
        choices=("candidate_a", "candidate_b"),
        default=[],
    )
    parser.add_argument("--selection-note")
    parser.add_argument("--selection-note-variant", action="append", default=[])
    parser.add_argument("--prompt-name", default="Script contextual extraction prompt")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--prompt-template-variant", action="append", default=[])
    parser.add_argument("--static-context-name", default="Script static context")
    parser.add_argument("--static-context-json", default="{}")
    parser.add_argument("--static-context-json-variant", action="append", default=[])
    parser.add_argument("--schema-name", default="Script contextual extraction schema")
    parser.add_argument("--schema-json", default=json.dumps(DEFAULT_SCHEMA))
    parser.add_argument("--schema-json-variant", action="append", default=[])
    parser.add_argument("--model-provider", default="local")
    parser.add_argument("--model-provider-variant", action="append", default=[])
    parser.add_argument("--model", default="echo")
    parser.add_argument("--model-variant", action="append", default=[])
    parser.add_argument("--model-capability", action="append", default=None)
    parser.add_argument("--model-parameters-json", default="{}")
    parser.add_argument("--model-parameters-json-variant", action="append", default=[])
    parser.add_argument("--credential-ref")
    parser.add_argument("--endpoint-ref")
    parser.add_argument("--policy-name", default="Script contextual extraction policy")
    parser.add_argument(
        "--policy-type",
        choices=("stateless", "accumulating", "sliding_window", "custom"),
        default="accumulating",
    )
    parser.add_argument(
        "--policy-type-variant",
        action="append",
        choices=("stateless", "accumulating", "sliding_window", "custom"),
        default=[],
    )
    parser.add_argument("--policy-settings-json", default="{}")
    parser.add_argument("--policy-settings-json-variant", action="append", default=[])
    parser.add_argument("--result-key", default="record")
    parser.add_argument(
        "--export-format",
        choices=("json", "jsonl", "csv"),
        default="json",
    )
    parser.add_argument(
        "--export-format-variant",
        action="append",
        choices=("json", "jsonl", "csv"),
        default=[],
    )
    parser.add_argument("--workflow-name", default="Script extraction experiment workflow")
    parser.add_argument("--experiment-name", default="Script extraction experiment")
    parser.add_argument("--max-node-runs-per-variant", type=int, default=100)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    validate_page_sequence_arguments(parser, args)
    comparison_variant_values = (
        args.comparison_ocr_engine_variant,
        args.comparison_engine_config_json_variant,
        args.candidate_a_label_variant,
        args.candidate_b_label_variant,
        args.selected_candidate_variant,
        args.selection_note_variant,
    )
    if args.comparison_ocr_engine is None and any(comparison_variant_values):
        parser.error("comparison variants require --comparison-ocr-engine")

    variant_values = (
        args.ocr_engine_variant,
        args.engine_config_json_variant,
        *comparison_variant_values,
        args.prompt_template_variant,
        args.static_context_json_variant,
        args.schema_json_variant,
        args.model_provider_variant,
        args.model_variant,
        args.model_parameters_json_variant,
        args.policy_type_variant,
        args.policy_settings_json_variant,
        args.export_format_variant,
    )
    if not any(variant_values):
        parser.error("at least one --*-variant option is required")
    return args


def _workflow_template_id(args: argparse.Namespace) -> str:
    if args.comparison_ocr_engine is None:
        return "contextual-extraction"
    return "ocr-compare-contextual-extraction"


def _workflow_config(args: argparse.Namespace) -> JsonObject:
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

    config: JsonObject = {
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
            "json_schema": json_object_config(args.schema_json, "--schema-json"),
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
        config["ocr"] = {
            "engine": args.ocr_engine,
            "language_hints": args.language_hint,
            "engine_config": engine_config(args.engine_config_json),
        }
    else:
        config["ocr_a"] = {
            "engine": args.ocr_engine,
            "language_hints": args.language_hint,
            "engine_config": engine_config(args.engine_config_json),
        }
        config["ocr_b"] = {
            "engine": args.comparison_ocr_engine,
            "language_hints": args.language_hint,
            "engine_config": engine_config(args.comparison_engine_config_json),
        }
        config["compare"] = {
            "candidate_a_label": args.candidate_a_label,
            "candidate_b_label": args.candidate_b_label,
        }
        config["select"] = {
            "selected_candidate": args.selected_candidate,
            "decision_note": args.selection_note,
        }
    return config


def _parameter_presets(args: argparse.Namespace) -> list[JsonObject]:
    presets: list[JsonObject] = []
    if args.comparison_ocr_engine is None:
        _append_preset(presets, "ocr_engine", "ocr", args.ocr_engine_variant)
        _append_json_preset(
            presets,
            "ocr_engine_config",
            "ocr",
            args.engine_config_json_variant,
            "--engine-config-json-variant",
        )
    else:
        _append_preset(
            presets,
            "ocr_engine",
            "ocr_a",
            args.ocr_engine_variant,
            name="ocr_a_engine",
        )
        _append_json_preset(
            presets,
            "ocr_engine_config",
            "ocr_a",
            args.engine_config_json_variant,
            "--engine-config-json-variant",
            name="ocr_a_engine_config",
        )
        _append_preset(
            presets,
            "ocr_engine",
            "ocr_b",
            args.comparison_ocr_engine_variant,
            name="ocr_b_engine",
        )
        _append_json_preset(
            presets,
            "ocr_engine_config",
            "ocr_b",
            args.comparison_engine_config_json_variant,
            "--comparison-engine-config-json-variant",
            name="ocr_b_engine_config",
        )
        _append_preset(
            presets,
            "ocr_candidate_a_label",
            "compare",
            args.candidate_a_label_variant,
        )
        _append_preset(
            presets,
            "ocr_candidate_b_label",
            "compare",
            args.candidate_b_label_variant,
        )
        _append_preset(
            presets,
            "ocr_selected_candidate",
            "select",
            args.selected_candidate_variant,
        )
        _append_preset(
            presets,
            "ocr_selection_note",
            "select",
            args.selection_note_variant,
        )
    _append_preset(
        presets,
        "prompt_template",
        "prompt",
        args.prompt_template_variant,
    )
    _append_json_preset(
        presets,
        "static_context",
        "context",
        args.static_context_json_variant,
        "--static-context-json-variant",
    )
    _append_json_preset(
        presets,
        "extraction_schema",
        "schema",
        args.schema_json_variant,
        "--schema-json-variant",
    )
    _append_preset(presets, "model_provider", "model", args.model_provider_variant)
    _append_preset(presets, "model_name", "model", args.model_variant)
    _append_json_preset(
        presets,
        "model_parameters",
        "model",
        args.model_parameters_json_variant,
        "--model-parameters-json-variant",
    )
    _append_preset(presets, "input_policy_type", "policy", args.policy_type_variant)
    _append_json_preset(
        presets,
        "input_policy_settings",
        "policy",
        args.policy_settings_json_variant,
        "--policy-settings-json-variant",
    )
    _append_preset(presets, "export_format", "export", args.export_format_variant)
    return presets


def _append_preset(
    presets: list[JsonObject],
    kind: str,
    node_id: str,
    values: list[object],
    *,
    name: str | None = None,
) -> None:
    if not values:
        return
    preset: JsonObject = {"kind": kind, "node_id": node_id, "values": values}
    if name is not None:
        preset["name"] = name
    presets.append(preset)


def _append_json_preset(
    presets: list[JsonObject],
    kind: str,
    node_id: str,
    raw_values: list[str],
    option_name: str,
    *,
    name: str | None = None,
) -> None:
    if not raw_values:
        return
    preset: JsonObject = {
        "kind": kind,
        "node_id": node_id,
        "values": [
            json_object_config(raw_value, option_name) for raw_value in raw_values
        ],
    }
    if name is not None:
        preset["name"] = name
    presets.append(preset)


def _variant_summary(
    comparison_variant: JsonObject,
    execution: JsonObject,
    outputs: JsonObject,
    *,
    expected_page_count: int,
) -> JsonObject:
    variant_id = comparison_variant["variant_id"]
    document_payloads = _variant_json_payloads(
        outputs,
        variant_id,
        "extraction.document_result",
    )
    if len(document_payloads) != 1:
        raise RuntimeError(
            "Experiment variant "
            f"{comparison_variant['variant_key']} produced "
            f"{len(document_payloads)} extraction.document_result artifacts"
        )
    document_payload = document_payloads[0]
    if document_payload["page_count"] != expected_page_count:
        raise RuntimeError(
            "Experiment variant "
            f"{comparison_variant['variant_key']} document page_count "
            f"{document_payload['page_count']} does not match {expected_page_count}"
        )
    return {
        "variant_id": variant_id,
        "variant_key": comparison_variant["variant_key"],
        "parameter_values": comparison_variant["parameter_values"],
        "workflow_run_id": comparison_variant["workflow_run_id"],
        "workflow_run_status": comparison_variant["workflow_run_status"],
        "processed_node_run_ids": _processed_node_run_ids(execution, variant_id),
        "errors": _variant_errors(execution, variant_id),
        "artifact_counts": comparison_variant["artifact_counts"],
        "validation_error_count": comparison_variant["validation_error_count"],
        "total_duration_ms": comparison_variant["total_duration_ms"],
        "total_cost": comparison_variant["total_cost"],
        "metric_values": comparison_variant["metric_values"],
        "document_payload": document_payload,
        "record_payloads": _variant_json_payloads(
            outputs,
            variant_id,
            "extraction.record_result",
        ),
        "ocr_comparison_payloads": _variant_json_payloads(
            outputs,
            variant_id,
            "ocr.comparison_result",
        ),
        "model_input_payloads": _variant_json_payloads(
            outputs,
            variant_id,
            "model.input",
        ),
        "export_dataset_payloads": _variant_json_payloads(
            outputs,
            variant_id,
            "export.dataset",
        ),
        "export_dataset_texts": _variant_text_payloads(
            outputs,
            variant_id,
            "export.dataset",
        ),
        "trace_counts": _variant_trace_counts(outputs, variant_id),
    }


def _processed_node_run_ids(
    execution: JsonObject,
    variant_id: object,
) -> list[object]:
    for variant in object_list_field(execution, "variants"):
        if variant["variant_id"] == variant_id:
            processed_ids = variant["processed_node_run_ids"]
            if not isinstance(processed_ids, list):
                raise RuntimeError("processed_node_run_ids is not a list")
            return processed_ids
    raise RuntimeError(f"Variant {variant_id} missing from execution response")


def _variant_errors(execution: JsonObject, variant_id: object) -> list[object]:
    for variant in object_list_field(execution, "variants"):
        if variant["variant_id"] == variant_id:
            errors = variant["errors"]
            if not isinstance(errors, list):
                raise RuntimeError("errors is not a list")
            return errors
    raise RuntimeError(f"Variant {variant_id} missing from execution response")


def _variant_json_payloads(
    outputs: JsonObject,
    variant_id: object,
    artifact_type: str,
) -> list[JsonObject]:
    payloads: list[JsonObject] = []
    for output in _variant_outputs(outputs, variant_id):
        artifact = object_field(output, "artifact")
        if artifact["artifact_type"] != artifact_type:
            continue
        payload = object_field(output, "payload")
        if payload["error"] is not None:
            raise RuntimeError(f"{artifact_type} output payload failed: {payload['error']}")
        json_payload = payload["json_payload"]
        if not isinstance(json_payload, dict):
            continue
        payloads.append(json_payload)
    return payloads


def _variant_text_payloads(
    outputs: JsonObject,
    variant_id: object,
    artifact_type: str,
) -> list[str]:
    payloads: list[str] = []
    for output in _variant_outputs(outputs, variant_id):
        artifact = object_field(output, "artifact")
        if artifact["artifact_type"] != artifact_type:
            continue
        payload = object_field(output, "payload")
        if payload["error"] is not None:
            raise RuntimeError(f"{artifact_type} output payload failed: {payload['error']}")
        text = payload["text"]
        if isinstance(text, str):
            payloads.append(text)
    return payloads


def _variant_outputs(outputs: JsonObject, variant_id: object) -> list[JsonObject]:
    for variant in object_list_field(outputs, "variants"):
        if variant["variant_id"] != variant_id:
            continue
        output_bundle = object_field(variant, "output_bundle")
        return object_list_field(output_bundle, "artifacts")
    raise RuntimeError(f"Variant {variant_id} missing from experiment outputs")


def _variant_trace_counts(outputs: JsonObject, variant_id: object) -> JsonObject:
    for variant in object_list_field(outputs, "variants"):
        if variant["variant_id"] != variant_id:
            continue
        output_bundle = object_field(variant, "output_bundle")
        input_assembly_trace_count = 0
        invocation_trace_count = 0
        for trace_bundle in object_list_field(output_bundle, "traces"):
            input_assembly_trace_count += len(
                object_list_field(trace_bundle, "input_assembly_traces")
            )
            invocation_trace_count += len(
                object_list_field(trace_bundle, "invocation_traces")
            )
        return {
            "input_assembly": input_assembly_trace_count,
            "invocation": invocation_trace_count,
        }
    raise RuntimeError(f"Variant {variant_id} missing from experiment outputs")


if __name__ == "__main__":
    main()
