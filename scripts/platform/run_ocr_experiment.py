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
        baseline_engine = args.engine[0]

        materialized = client.request_object(
            "POST",
            "/v1/workflow-templates/ocr-pages/materialize",
            {
                "name": args.workflow_name,
                "config": {
                    "ocr": {
                        "engine": baseline_engine,
                        "language_hints": args.language_hint,
                        "engine_config": engine_config(args.engine_config_json),
                    }
                },
                "metadata": {
                    "runner": "scripts/platform/run_ocr_experiment.py",
                    "source_sequence_id": str(sequence["id"]),
                },
                "change_note": "Created by scripts/platform/run_ocr_experiment.py",
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
                "parameter_presets": [
                    {
                        "kind": "ocr_engine",
                        "node_id": "ocr",
                        "values": args.engine,
                    }
                ],
                "input_artifact_sequence_refs": [
                    {
                        "sequence_id": sequence["id"],
                        "artifact_type": PAGE_IMAGE_ARTIFACT_TYPE,
                        "schema_version": PAGE_IMAGE_SCHEMA_VERSION,
                    }
                ],
                "metadata": {
                    "runner": "scripts/platform/run_ocr_experiment.py",
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
            query={"artifact_type": "ocr.page_result", "include_payloads": "true"},
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
        description="Run an OCR engine experiment through the Notarius HTTP API."
    )
    parser.add_argument("--api-base-url", default=api_base_url_default())
    parser.add_argument(
        "--engine",
        action="append",
        choices=("local.tesseract", "local.text", "mistral.ocr"),
        default=None,
        help="OCR engine variant. Repeat to compare multiple engines.",
    )
    add_page_sequence_arguments(parser)
    parser.add_argument("--language-hint", action="append", default=[])
    parser.add_argument(
        "--engine-config-json",
        default="{}",
        help=(
            "JSON object passed to every OCR engine. For mistral.ocr this can "
            "include model, base_url, timeout_seconds, include_blocks, and "
            "api_key_env_var."
        ),
    )
    parser.add_argument("--workflow-name", default="Script OCR experiment workflow")
    parser.add_argument("--experiment-name", default="Script OCR experiment")
    parser.add_argument("--max-node-runs-per-variant", type=int, default=100)
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    validate_page_sequence_arguments(parser, args)
    if args.engine is None or len(args.engine) < 2:
        parser.error("at least two --engine values are required")
    return args


def _variant_summary(
    comparison_variant: JsonObject,
    execution: JsonObject,
    outputs: JsonObject,
    *,
    expected_page_count: int,
) -> JsonObject:
    variant_id = comparison_variant["variant_id"]
    ocr_payloads = _variant_ocr_payloads(outputs, variant_id)
    if len(ocr_payloads) != expected_page_count:
        raise RuntimeError(
            "Experiment variant "
            f"{comparison_variant['variant_key']} produced {len(ocr_payloads)} "
            f"OCR payloads for {expected_page_count} source pages"
        )
    return {
        "variant_id": variant_id,
        "variant_key": comparison_variant["variant_key"],
        "parameter_values": comparison_variant["parameter_values"],
        "engine": object_field(comparison_variant, "parameter_values")["engine"],
        "workflow_run_id": comparison_variant["workflow_run_id"],
        "workflow_run_status": comparison_variant["workflow_run_status"],
        "processed_node_run_ids": _processed_node_run_ids(execution, variant_id),
        "errors": _variant_errors(execution, variant_id),
        "artifact_counts": comparison_variant["artifact_counts"],
        "output_sequence_ids": _variant_output_sequence_ids(outputs, variant_id),
        "validation_error_count": comparison_variant["validation_error_count"],
        "total_duration_ms": comparison_variant["total_duration_ms"],
        "total_cost": comparison_variant["total_cost"],
        "metric_values": comparison_variant["metric_values"],
        "ocr_payloads": ocr_payloads,
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


def _variant_output_sequence_ids(outputs: JsonObject, variant_id: object) -> list[object]:
    for variant in object_list_field(outputs, "variants"):
        if variant["variant_id"] != variant_id:
            continue
        output_bundle = object_field(variant, "output_bundle")
        return [
            sequence["id"]
            for sequence in object_list_field(output_bundle, "artifact_sequences")
        ]
    raise RuntimeError(f"Variant {variant_id} missing from experiment outputs")


def _variant_ocr_payloads(outputs: JsonObject, variant_id: object) -> list[JsonObject]:
    for variant in object_list_field(outputs, "variants"):
        if variant["variant_id"] != variant_id:
            continue
        output_bundle = object_field(variant, "output_bundle")
        ocr_payloads: list[JsonObject] = []
        for output in object_list_field(output_bundle, "artifacts"):
            payload = output["payload"]
            if not isinstance(payload, dict):
                raise RuntimeError("OCR output artifact payload is not an object")
            payload_error = payload["error"]
            if payload_error is not None:
                raise RuntimeError(f"OCR output artifact payload failed: {payload_error}")
            ocr_payload = payload["json_payload"]
            if not isinstance(ocr_payload, dict):
                raise RuntimeError("OCR output artifact payload JSON is not an object")
            ocr_payloads.append(ocr_payload)
        return ocr_payloads
    raise RuntimeError(f"Variant {variant_id} missing from experiment outputs")


if __name__ == "__main__":
    main()
