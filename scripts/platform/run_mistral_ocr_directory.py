import json
import os
from dataclasses import dataclass
from pathlib import Path

from ocr_script_support import (
    PAGE_IMAGE_ARTIFACT_TYPE,
    PAGE_IMAGE_SCHEMA_VERSION,
    ApiClient,
    JsonObject,
    api_base_url_default,
    image_content_type,
    object_field,
    object_list_field,
    require_page_image_sequence,
    sequence_page_count,
)


IMAGE_DIRECTORY = Path(os.getenv("NOTARIUS_MISTRAL_OCR_IMAGE_DIR", "data/ocr-pages"))
OUTPUT_JSON_PATH = Path(
    os.getenv("NOTARIUS_MISTRAL_OCR_OUTPUT_JSON", "data/outputs/mistral-ocr-result.json")
)
PROJECT_NAME = os.getenv("NOTARIUS_MISTRAL_OCR_PROJECT_NAME", "Mistral OCR project")
SOURCE_NAME = os.getenv("NOTARIUS_MISTRAL_OCR_SOURCE_NAME", "Mistral OCR pages")
LANGUAGE_HINTS = tuple(
    hint.strip()
    for hint in os.getenv("NOTARIUS_MISTRAL_OCR_LANGUAGE_HINTS", "").split(",")
    if hint.strip() != ""
)
MISTRAL_ENGINE_CONFIG: JsonObject = {
    "api_key_env_var": os.getenv(
        "NOTARIUS_MISTRAL_OCR_API_KEY_ENV_VAR",
        "MISTRAL_API_KEY",
    ),
    "model": os.getenv("NOTARIUS_MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
    "include_blocks": ["text"],
}

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".webp")


@dataclass(frozen=True, slots=True)
class MistralDirectoryOcrConfig:
    api_base_url: str
    image_directory: Path
    output_json_path: Path | None
    project_name: str
    source_name: str
    language_hints: tuple[str, ...]
    engine_config: JsonObject


def main() -> None:
    result = run_mistral_ocr_directory(default_config())
    print(json.dumps(result, indent=2))


def default_config() -> MistralDirectoryOcrConfig:
    engine_config = dict(MISTRAL_ENGINE_CONFIG)
    raw_engine_config_json = os.getenv("NOTARIUS_MISTRAL_OCR_ENGINE_CONFIG_JSON")
    if raw_engine_config_json is not None:
        decoded_engine_config = json.loads(raw_engine_config_json)
        if not isinstance(decoded_engine_config, dict):
            raise RuntimeError(
                "NOTARIUS_MISTRAL_OCR_ENGINE_CONFIG_JSON must decode to an object"
            )
        engine_config.update(decoded_engine_config)

    return MistralDirectoryOcrConfig(
        api_base_url=api_base_url_default(),
        image_directory=IMAGE_DIRECTORY,
        output_json_path=OUTPUT_JSON_PATH,
        project_name=PROJECT_NAME,
        source_name=SOURCE_NAME,
        language_hints=LANGUAGE_HINTS,
        engine_config=engine_config,
    )


def run_mistral_ocr_directory(config: MistralDirectoryOcrConfig) -> JsonObject:
    image_paths = directory_image_paths(config.image_directory)
    client = ApiClient(config.api_base_url)
    project = client.request_object(
        "POST",
        "/v1/projects",
        {"name": config.project_name},
    )
    uploaded = client.request_multipart(
        f"/v1/projects/{project['id']}/sources/images",
        fields={"name": config.source_name},
        files=[
            ("files", image_path, image_content_type(image_path))
            for image_path in image_paths
        ],
    )
    source = object_field(uploaded, "source")
    sequence = object_field(uploaded, "sequence")
    require_page_image_sequence(sequence)

    launch = client.request_object(
        "POST",
        "/v1/workflow-templates/ocr-pages/launch",
        {
            "name": "Mistral OCR directory",
            "config": {
                "ocr": {
                    "engine": "mistral.ocr",
                    "language_hints": list(config.language_hints),
                    "engine_config": config.engine_config,
                },
                "execution_planning": "concrete_map",
            },
            "input_artifact_sequence_refs": [
                {
                    "sequence_id": sequence["id"],
                    "artifact_type": PAGE_IMAGE_ARTIFACT_TYPE,
                    "schema_version": PAGE_IMAGE_SCHEMA_VERSION,
                }
            ],
            "metadata": {
                "runner": "scripts/platform/run_mistral_ocr_directory.py",
                "source_sequence_id": str(sequence["id"]),
                "image_directory": str(config.image_directory),
            },
            "change_note": "Created by scripts/platform/run_mistral_ocr_directory.py",
        },
    )
    workflow = object_field(launch, "workflow_definition")
    version = object_field(launch, "workflow_version")
    run = object_field(launch, "workflow_run")

    page_count = sequence_page_count(sequence)
    execution = client.request_object(
        "POST",
        f"/v1/workflow-runs/{run['id']}/execute",
        {"max_node_runs": page_count + 1},
    )
    errors = execution["errors"]
    if errors != []:
        raise RuntimeError(f"Workflow run {run['id']} failed: {errors}")

    summary = client.request_object("GET", f"/v1/workflow-runs/{run['id']}/summary")
    page_outputs = client.request_object(
        "GET",
        f"/v1/workflow-runs/{run['id']}/outputs",
        query={"artifact_type": "ocr.page_result", "include_payloads": "true"},
    )
    ocr_payloads = json_payloads(page_outputs)
    document_outputs = client.request_object(
        "GET",
        f"/v1/workflow-runs/{run['id']}/outputs",
        query={"artifact_type": "ocr.document_result", "include_payloads": "true"},
    )
    document_payloads = json_payloads(document_outputs)
    if len(ocr_payloads) != page_count:
        raise RuntimeError(
            f"Mistral OCR produced {len(ocr_payloads)} page payloads for "
            f"{page_count} input pages"
        )
    if len(document_payloads) != 1:
        raise RuntimeError(
            f"Mistral OCR produced {len(document_payloads)} document payloads"
        )

    result: JsonObject = {
        "project_id": project["id"],
        "source_id": source["id"],
        "source_kind": "image_directory",
        "source_page_count": page_count,
        "source_sequence_id": sequence["id"],
        "image_directory": str(config.image_directory),
        "image_paths": [str(path) for path in image_paths],
        "workflow_id": workflow["id"],
        "workflow_version_id": version["id"],
        "workflow_run_id": run["id"],
        "workflow_template_id": object_field(launch, "template")["id"],
        "execution_planning": "concrete_map",
        "workflow_run_status": object_field(summary, "workflow_run")["status"],
        "processed_node_run_ids": execution["processed_node_run_ids"],
        "artifact_counts": summary["artifact_counts"],
        "ocr_payloads": ocr_payloads,
        "ocr_document_payload": document_payloads[0],
    }
    if config.output_json_path is not None:
        write_json_file(config.output_json_path, result)
    return result


def directory_image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise RuntimeError(f"Image directory does not exist: {directory}")
    image_paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    image_paths.sort(key=lambda path: path.name.lower())
    if not image_paths:
        suffixes = ", ".join(IMAGE_SUFFIXES)
        raise RuntimeError(f"Image directory {directory} contains no {suffixes} files")
    return image_paths


def json_payloads(outputs: JsonObject) -> list[JsonObject]:
    payloads: list[JsonObject] = []
    for output in object_list_field(outputs, "artifacts"):
        payload = output["payload"]
        if not isinstance(payload, dict):
            raise RuntimeError("Output artifact payload is not an object")
        payload_error = payload["error"]
        if payload_error is not None:
            raise RuntimeError(f"Output artifact payload failed: {payload_error}")
        json_payload = payload["json_payload"]
        if not isinstance(json_payload, dict):
            raise RuntimeError("Output artifact payload JSON is not an object")
        payloads.append(json_payload)
    return payloads


def write_json_file(path: Path, value: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
