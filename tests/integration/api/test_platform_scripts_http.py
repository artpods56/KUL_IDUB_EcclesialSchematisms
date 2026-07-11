import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request
from urllib.request import urlopen

import fitz
import pytest
import uvicorn

from notarius_api import dependencies as api_deps
from notarius_api.main import app
from notarius_persistence.adapters.in_memory import InMemoryDataStore


@pytest.fixture
def live_api_base_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[str]:
    store = InMemoryDataStore()
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("NOTARIUS_ARTIFACT_PAYLOAD_DIR", str(tmp_path / "artifacts"))
    app.dependency_overrides[api_deps.get_store] = lambda: store
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_for_api(base_url)
        yield base_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        app.dependency_overrides.pop(api_deps.get_store, None)


def test_run_debug_workflow_script_over_real_http(
    live_api_base_url: str,
) -> None:
    result = _run_script(
        live_api_base_url,
        ["scripts/platform/run_debug_workflow.py"],
    )

    assert result["workflow_run_status"] == "succeeded"
    assert len(_list_field(result, "processed_node_run_ids")) == 1
    assert result["artifact_counts"] == {"debug.text": 1}


def test_watch_workflow_run_events_script_over_real_http(
    live_api_base_url: str,
) -> None:
    workflow_result = _run_script(
        live_api_base_url,
        ["scripts/platform/run_debug_workflow.py"],
    )

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/watch_workflow_run_events.py",
            str(workflow_result["workflow_run_id"]),
        ],
    )

    timeline = _dict_field(result, "timeline")
    events = _list_field(timeline, "events")
    assert result["workflow_run_id"] == workflow_result["workflow_run_id"]
    assert result["poll_count"] == 1
    assert _dict_field(timeline, "workflow_run")["status"] == "succeeded"
    assert "workflow_run" in {event["event_kind"] for event in events}
    assert "node_run" in {event["event_kind"] for event in events}
    assert "artifact" in {event["event_kind"] for event in events}


def test_manage_outbox_script_lists_pending_messages_over_real_http(
    live_api_base_url: str,
) -> None:
    _run_script(live_api_base_url, ["scripts/platform/run_debug_workflow.py"])

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_outbox.py",
            "list",
            "--status",
            "pending",
            "--subject-prefix",
            "events.workflow_run.",
            "--limit",
            "1",
        ],
    )

    messages = _list_field(result, "messages")
    assert result["status"] == "pending"
    assert result["subject_prefix"] == "events.workflow_run."
    assert result["limit"] == 1
    assert len(messages) == 1
    assert messages[0]["subject"].startswith("events.workflow_run.")


def test_manage_outbox_script_summarizes_dlq_messages_over_real_http(
    live_api_base_url: str,
) -> None:
    result = _run_script(
        live_api_base_url,
        ["scripts/platform/manage_outbox.py", "dlq-summary"],
    )

    assert _list_field(result, "summaries") == []


def test_manage_outbox_script_previews_cleanup_over_real_http(
    live_api_base_url: str,
) -> None:
    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_outbox.py",
            "cleanup",
            "--older-than",
            "2100-01-01T00:00:00+00:00",
        ],
    )

    cleanup = _dict_field(result, "cleanup")
    assert cleanup["dry_run"] is True
    assert cleanup["matched_count"] == 0
    assert cleanup["deleted_count"] == 0
    assert cleanup["messages"] == []


def test_manage_outbox_script_archives_cleanup_preview_before_execute(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "archives" / "outbox-cleanup.json"

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_outbox.py",
            "cleanup",
            "--older-than",
            "2100-01-01T00:00:00+00:00",
            "--archive-path",
            str(archive_path),
            "--execute",
        ],
    )

    cleanup = _dict_field(result, "cleanup")
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    assert cleanup["dry_run"] is False
    assert cleanup["matched_count"] == 0
    assert cleanup["deleted_count"] == 0
    assert result["archive_path"] == str(archive_path)
    assert result["archived_count"] == 0
    assert archive["request"]["dry_run"] is True
    assert archive["request"]["statuses"] == ["published", "failed"]
    assert archive["cleanup"]["dry_run"] is True
    assert archive["cleanup"]["messages"] == []


def test_manage_experiment_script_reruns_failed_variants_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)
    workflow = _request_object(
        live_api_base_url,
        "POST",
        "/v1/workflows",
        {
            "name": "Script-managed failing OCR workflow",
            "nodes": [
                {
                    "id": "ocr",
                    "operator_id": "ocr.extract_pages",
                    "operator_version": "1.0.0",
                    "config": {"engine": "missing.ocr"},
                }
            ],
            "declared_inputs": [
                {
                    "name": "pages",
                    "artifact_type": "source.page_image",
                    "schema_version": 1,
                    "sequence": True,
                    "required": True,
                    "description": None,
                }
            ],
        },
    )
    version = _request_object(
        live_api_base_url,
        "POST",
        f"/v1/workflows/{workflow['id']}/versions",
        {"change_note": "Create failing OCR experiment version"},
    )
    experiment = _request_object(
        live_api_base_url,
        "POST",
        "/v1/experiments",
        {
            "name": "Script-managed failing OCR experiment",
            "workflow_version_id": version["id"],
            "input_artifact_sequence_refs": [
                {
                    "sequence_id": source_result["source_sequence_id"],
                    "artifact_type": "source.page_image",
                    "schema_version": 1,
                }
            ],
        },
    )
    execution = _request_object(
        live_api_base_url,
        "POST",
        f"/v1/experiments/{experiment['id']}/execute",
        {"max_node_runs_per_variant": 10},
    )
    failed_variant = _list_field(execution, "variants")[0]
    old_run_id = failed_variant["workflow_run_id"]

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_experiment.py",
            "rerun-failed",
            str(experiment["id"]),
        ],
    )

    rerun = _dict_field(result, "rerun")
    rerun_experiment = _dict_field(rerun, "experiment")
    rerun_variants = _list_field(rerun, "variants")
    rerun_variant = rerun_variants[0]
    comparison = _request_object(
        live_api_base_url,
        "GET",
        f"/v1/experiments/{experiment['id']}/comparison",
    )

    assert _list_field(failed_variant, "errors") != []
    assert rerun_experiment["status"] == "queued"
    assert len(rerun_variants) == 1
    assert rerun_variant["previous_workflow_run_id"] == old_run_id
    assert rerun_variant["workflow_run_id"] != old_run_id
    assert rerun_experiment["variants"][0]["workflow_run_id"] == (
        rerun_variant["workflow_run_id"]
    )
    assert rerun_experiment["variants"][0]["metadata"] == {
        "previous_workflow_run_ids": [old_run_id],
        "rerun_count": 1,
        "rerun_of_workflow_run_id": old_run_id,
    }
    comparison_variants = _list_field(comparison, "variants")
    assert comparison_variants[0]["workflow_run_id"] == rerun_variant["workflow_run_id"]
    assert comparison_variants[0]["workflow_run_status"] == "queued"


def test_manage_experiment_script_writes_comparison_csv_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)
    experiment_result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_experiment.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--prompt-template-variant",
            "CSV A: {{ CURRENT_PAGE_TEXT }}",
            "--prompt-template-variant",
            "CSV B: {{ CURRENT_PAGE_TEXT }}",
        ],
    )
    output_path = tmp_path / "comparison.csv"

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_experiment.py",
            "comparison",
            str(experiment_result["experiment_id"]),
            "--output-format",
            "csv",
            "--output-path",
            str(output_path),
        ],
    )

    csv_text = output_path.read_text(encoding="utf-8")
    assert result == {"output_path": str(output_path)}
    assert "variant_key,workflow_run_id,workflow_run_status" in csv_text
    assert "param.template" in csv_text
    assert "metric.summary.invocation_count" in csv_text
    assert "CSV A: {{ CURRENT_PAGE_TEXT }}" in csv_text
    assert "CSV B: {{ CURRENT_PAGE_TEXT }}" in csv_text
    assert "succeeded" in csv_text


def test_manage_experiment_script_writes_filtered_outputs_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)
    experiment_result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_experiment.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--export-format-variant",
            "json",
            "--export-format-variant",
            "csv",
        ],
    )
    output_path = tmp_path / "experiment-outputs.json"

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_experiment.py",
            "outputs",
            str(experiment_result["experiment_id"]),
            "--artifact-type",
            "export.dataset",
            "--include-payloads",
            "--include-text-payloads",
            "--output-path",
            str(output_path),
        ],
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    outputs = _dict_field(saved, "outputs")
    variants = _list_field(outputs, "variants")
    assert result == {"output_path": str(output_path)}
    assert len(variants) == 2
    for variant in variants:
        artifacts = _list_field(_dict_field(variant, "output_bundle"), "artifacts")
        assert len(artifacts) == 1
        artifact = _dict_field(artifacts[0], "artifact")
        assert artifact["artifact_type"] == "export.dataset"
    first_artifact = _list_field(
        _dict_field(variants[0], "output_bundle"),
        "artifacts",
    )[0]
    second_artifact = _list_field(
        _dict_field(variants[1], "output_bundle"),
        "artifacts",
    )[0]
    json_payload = _dict_field(_dict_field(first_artifact, "payload"), "json_payload")
    csv_text = _dict_field(second_artifact, "payload")["text"]
    assert json_payload["records"] == [
        {"text": "first page text", "page_number": 1},
        {"text": "second page text", "page_number": 2},
    ]
    assert csv_text == "text,page_number\nfirst page text,1\nsecond page text,2\n"


def test_manage_experiment_script_writes_variant_event_timelines_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    experiment_result = _run_script(
        live_api_base_url,
        ["scripts/platform/run_debug_experiment.py"],
    )
    output_path = tmp_path / "experiment-events.json"

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_experiment.py",
            "events",
            str(experiment_result["experiment_id"]),
            "--output-path",
            str(output_path),
        ],
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    experiment_events = _dict_field(saved, "experiment_events")
    variant_timelines = _list_field(experiment_events, "variant_timelines")
    assert result == {"output_path": str(output_path)}
    assert _dict_field(experiment_events, "experiment")["id"] == (
        experiment_result["experiment_id"]
    )
    assert len(variant_timelines) == 2
    for variant_timeline in variant_timelines:
        assert variant_timeline["variant_key"] in {"variant-0001", "variant-0002"}
        timeline = _dict_field(variant_timeline, "timeline")
        assert _dict_field(timeline, "workflow_run")["status"] == "succeeded"
        events = _list_field(timeline, "events")
        assert {"workflow_run", "node_run", "artifact"}.issubset(
            {event["event_kind"] for event in events}
        )


def test_manage_artifact_script_inspects_payload_and_lineage_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    definition = _request_object(
        live_api_base_url,
        "POST",
        "/v1/workflows",
        {"name": "Script artifact inspection", "nodes": []},
    )
    version = _request_object(
        live_api_base_url,
        "POST",
        f"/v1/workflows/{definition['id']}/versions",
        {"change_note": "Inspect artifact script"},
    )
    run = _request_object(
        live_api_base_url,
        "POST",
        "/v1/workflow-runs",
        {"workflow_version_id": version["id"]},
    )
    node_run = _request_object(
        live_api_base_url,
        "POST",
        f"/v1/workflow-runs/{run['id']}/node-runs",
        {
            "workflow_node_id": "export",
            "operator_id": "export.dataset",
            "operator_version": "1.0.0",
        },
    )
    input_artifact = _request_object(
        live_api_base_url,
        "POST",
        "/v1/artifacts/json",
        {
            "artifact_type": "extraction.document_result",
            "schema_version": 1,
            "workflow_run_id": run["id"],
            "payload": {"records": [{"name": "Alpha"}]},
        },
    )
    output_payload = {"records": [{"name": "Alpha", "page": 1}]}
    output_artifact = _request_object(
        live_api_base_url,
        "POST",
        "/v1/artifacts/json",
        {
            "artifact_type": "export.dataset",
            "schema_version": 1,
            "workflow_run_id": run["id"],
            "producer_node_run_id": node_run["id"],
            "producer_operator_id": "export.dataset",
            "producer_operator_version": "1.0.0",
            "input_artifact_ids": [input_artifact["id"]],
            "payload": output_payload,
        },
    )
    output_path = tmp_path / "artifact-inspection.json"

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/manage_artifact.py",
            "inspect",
            str(output_artifact["id"]),
            "--include-payload",
            "--include-lineage",
            "--output-path",
            str(output_path),
        ],
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    inspection = _dict_field(saved, "inspection")
    payload = _dict_field(inspection, "payload")
    lineage = _dict_field(inspection, "lineage")
    assert result == {"output_path": str(output_path)}
    assert _dict_field(inspection, "artifact")["id"] == output_artifact["id"]
    assert payload["json_payload"] == output_payload
    assert _dict_field(lineage, "root_artifact")["id"] == output_artifact["id"]
    assert {artifact["id"] for artifact in _list_field(lineage, "artifacts")} == {
        input_artifact["id"],
        output_artifact["id"],
    }


def test_validate_workflow_definition_script_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "name": "Validate debug workflow",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "validated from script"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_script(
        live_api_base_url,
        ["scripts/platform/validate_workflow_definition.py", str(workflow_path)],
    )

    assert result["valid"] is True
    assert result["errors"] == []
    assert result["node_count"] == 1
    assert result["edge_count"] == 0
    assert result["execution_order"] == ["emit"]
    execution_plan = _dict_field(result, "execution_plan")
    assert execution_plan["execution_order"] == ["emit"]
    assert len(_list_field(execution_plan, "root_node_run_ids")) == 1
    plan_nodes = _list_field(execution_plan, "nodes")
    assert plan_nodes[0]["workflow_node_id"] == "emit"
    assert plan_nodes[0]["execution_mode"] == "single"
    assert plan_nodes[0]["root"] is True
    assert plan_nodes[0]["leaf"] is True


def test_validate_workflow_definition_script_can_fail_on_invalid_graph(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "invalid-workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "name": "Invalid workflow",
                "nodes": [
                    {
                        "id": "missing",
                        "operator_id": "missing.operator",
                        "operator_version": "1.0.0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = _run_script_failure(
        live_api_base_url,
        [
            "scripts/platform/validate_workflow_definition.py",
            "--fail-on-invalid",
            str(workflow_path),
        ],
    )

    assert completed.returncode == 1
    parsed = json.loads(completed.stdout)
    assert parsed["valid"] is False
    assert "Unknown operator" in parsed["errors"][0]


def test_run_workflow_definition_script_runs_generic_graph_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    workflow_path = tmp_path / "generic-workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "name": "Generic script debug workflow",
                "nodes": [
                    {
                        "id": "emit",
                        "operator_id": "debug.emit_text",
                        "operator_version": "1.0.0",
                        "config": {"text": "generic runner text"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_workflow_definition.py",
            "--include-outputs",
            "--include-traces",
            "--output-artifact-type",
            "debug.text",
            str(workflow_path),
        ],
    )

    assert result["workflow_run_status"] == "succeeded"
    assert result["validation"]["valid"] is True
    assert result["validation"]["execution_order"] == ["emit"]
    assert result["validation"]["execution_plan"]["execution_order"] == ["emit"]
    assert len(_list_field(result, "processed_node_run_ids")) == 1
    assert result["artifact_counts"] == {"debug.text": 1}
    outputs = _dict_field(result, "outputs")
    output_artifacts = _list_field(outputs, "artifacts")
    assert output_artifacts[0]["artifact"]["artifact_type"] == "debug.text"
    traces = _list_field(outputs, "traces")
    assert traces[0]["invocation_traces"][0]["invocation_type"] == "debug.emit_text"


def test_run_ocr_workflow_script_images_local_text_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    first_page = tmp_path / "page-1.png"
    second_page = tmp_path / "page-2.png"
    first_page.write_text("first page text", encoding="utf-8")
    second_page.write_text("second page text", encoding="utf-8")

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_workflow.py",
            "--engine",
            "local.text",
            "--image",
            str(first_page),
            "--image",
            str(second_page),
        ],
    )

    ocr_payloads = _list_field(result, "ocr_payloads")
    assert result["source_kind"] == "images"
    assert result["source_page_count"] == 2
    assert isinstance(result["source_sequence_id"], str)
    assert result["workflow_template_id"] == "ocr-pages"
    assert result["workflow_run_status"] == "succeeded"
    assert result["artifact_counts"]["ocr.page_result"] == 2
    assert result["artifact_counts"]["ocr.document_result"] == 1
    assert _dict_field(result, "ocr_document_payload")["text"] == (
        "first page text\n\nsecond page text"
    )
    assert [payload["text"] for payload in ocr_payloads] == [
        "first page text",
        "second page text",
    ]


def test_run_ocr_workflow_script_concrete_map_images_local_text_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    first_page = tmp_path / "concrete-page-1.png"
    second_page = tmp_path / "concrete-page-2.png"
    first_page.write_text("first concrete page text", encoding="utf-8")
    second_page.write_text("second concrete page text", encoding="utf-8")

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_workflow.py",
            "--engine",
            "local.text",
            "--concrete-map",
            "--image",
            str(first_page),
            "--image",
            str(second_page),
        ],
    )

    ocr_payloads = _list_field(result, "ocr_payloads")
    assert result["execution_planning"] == "concrete_map"
    assert result["workflow_run_status"] == "succeeded"
    assert len(_list_field(result, "processed_node_run_ids")) == 3
    assert result["artifact_counts"]["ocr.page_result"] == 2
    assert result["artifact_counts"]["ocr.document_result"] == 1
    assert _dict_field(result, "ocr_document_payload")["text"] == (
        "first concrete page text\n\nsecond concrete page text"
    )
    assert [payload["text"] for payload in ocr_payloads] == [
        "first concrete page text",
        "second concrete page text",
    ]


def test_run_mistral_ocr_directory_script_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_dir = tmp_path / "mistral-pages"
    image_dir.mkdir()
    first_page = image_dir / "001-page.png"
    second_page = image_dir / "002-page.jpg"
    first_page.write_text("first image bytes", encoding="utf-8")
    second_page.write_text("second image bytes", encoding="utf-8")
    output_path = tmp_path / "outputs" / "mistral-result.json"
    fake_mistral_url, fake_mistral_server, fake_mistral_thread = (
        _start_fake_mistral_ocr_server()
    )
    monkeypatch.setenv("MISTRAL_TEST_API_KEY", "secret-token")
    try:
        completed = subprocess.run(
            [sys.executable, "scripts/platform/run_mistral_ocr_directory.py"],
            check=True,
            cwd=Path(__file__).parents[3],
            env={
                **os.environ,
                "NOTARIUS_API_BASE_URL": live_api_base_url,
                "NOTARIUS_MISTRAL_OCR_IMAGE_DIR": str(image_dir),
                "NOTARIUS_MISTRAL_OCR_OUTPUT_JSON": str(output_path),
                "NOTARIUS_MISTRAL_OCR_ENGINE_CONFIG_JSON": json.dumps(
                    {
                        "api_key_env_var": "MISTRAL_TEST_API_KEY",
                        "base_url": fake_mistral_url,
                        "model": "mistral-test",
                        "include_blocks": ["text"],
                    }
                ),
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        fake_mistral_server.shutdown()
        fake_mistral_server.server_close()
        fake_mistral_thread.join(timeout=10)

    result = json.loads(completed.stdout)
    written_result = json.loads(output_path.read_text(encoding="utf-8"))
    ocr_payloads = _list_field(result, "ocr_payloads")

    assert written_result == result
    assert result["source_kind"] == "image_directory"
    assert result["source_page_count"] == 2
    assert result["execution_planning"] == "concrete_map"
    assert result["workflow_run_status"] == "succeeded"
    assert len(_list_field(result, "processed_node_run_ids")) == 3
    assert result["artifact_counts"]["ocr.page_result"] == 2
    assert result["artifact_counts"]["ocr.document_result"] == 1
    assert [payload["text"] for payload in ocr_payloads] == [
        "fake mistral page 1",
        "fake mistral page 2",
    ]
    assert _dict_field(result, "ocr_document_payload")["text"] == (
        "fake mistral page 1\n\nfake mistral page 2"
    )
    assert [
        _dict_field(payload, "runtime")["model"] for payload in ocr_payloads
    ] == [
        "mistral-test",
        "mistral-test",
    ]
    assert [
        request_payload["model"]
        for request_payload in _FakeMistralOcrHandler.request_payloads
    ] == [
        "mistral-test",
        "mistral-test",
    ]


def test_run_ocr_workflow_script_pdf_local_text_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "pages.pdf"
    pdf_path.write_bytes(_pdf_bytes(["first pdf page", "second pdf page"]))

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_workflow.py",
            "--engine",
            "local.text",
            "--pdf",
            str(pdf_path),
        ],
    )

    ocr_payloads = _list_field(result, "ocr_payloads")
    assert result["source_kind"] == "pdf"
    assert result["source_page_count"] == 2
    assert isinstance(result["source_sequence_id"], str)
    assert result["workflow_template_id"] == "ocr-pages"
    assert result["workflow_run_status"] == "succeeded"
    assert result["artifact_counts"]["ocr.page_result"] == 2
    assert result["artifact_counts"]["ocr.document_result"] == 1
    assert _dict_field(result, "ocr_document_payload")["page_count"] == 2
    assert [payload["page_number"] for payload in ocr_payloads] == [1, 2]


def test_run_ocr_workflow_script_reuses_existing_source_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    first_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    second_result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_workflow.py",
            "--engine",
            "local.text",
            "--source-id",
            str(first_result["source_id"]),
        ],
    )

    assert second_result["source_kind"] == "existing_source"
    assert second_result["source_id"] == first_result["source_id"]
    assert second_result["source_sequence_id"] == first_result["source_sequence_id"]
    assert second_result["workflow_template_id"] == "ocr-pages"
    assert second_result["source_page_count"] == first_result["source_page_count"]
    assert [payload["text"] for payload in _list_field(second_result, "ocr_payloads")] == [
        "first page text",
        "second page text",
    ]


def test_run_ocr_workflow_script_reuses_existing_sequence_over_real_http(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    first_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    second_result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_workflow.py",
            "--engine",
            "local.text",
            "--sequence-id",
            str(first_result["source_sequence_id"]),
        ],
    )

    assert second_result["source_kind"] == "existing_sequence"
    assert second_result["source_id"] == first_result["source_id"]
    assert second_result["project_id"] == first_result["project_id"]
    assert second_result["source_sequence_id"] == first_result["source_sequence_id"]
    assert second_result["workflow_template_id"] == "ocr-pages"
    assert second_result["source_page_count"] == first_result["source_page_count"]
    assert [payload["text"] for payload in _list_field(second_result, "ocr_payloads")] == [
        "first page text",
        "second page text",
    ]


def test_run_ocr_workflow_script_rejects_ambiguous_input_modes(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    page_path = tmp_path / "page-1.png"
    pdf_path = tmp_path / "pages.pdf"
    page_path.write_text("page text", encoding="utf-8")
    pdf_path.write_bytes(_pdf_bytes(["pdf text"]))

    failures = [
        (
            [
                "scripts/platform/run_ocr_workflow.py",
                "--pdf",
                str(pdf_path),
                "--source-id",
                "source-1",
            ],
            "--source-id cannot be combined with --image or --pdf",
        ),
        (
            [
                "scripts/platform/run_ocr_workflow.py",
                "--image",
                str(page_path),
                "--source-id",
                "source-1",
            ],
            "--source-id cannot be combined with --image or --pdf",
        ),
        (
            [
                "scripts/platform/run_ocr_workflow.py",
                "--source-id",
                "source-1",
                "--sequence-id",
                "sequence-1",
            ],
            "--source-id cannot be combined with --sequence-id",
        ),
    ]

    for args, expected_error in failures:
        completed = _run_script_failure(live_api_base_url, args)
        assert completed.returncode == 2
        assert expected_error in completed.stderr


def test_run_ocr_workflow_script_source_mode_rejects_ambiguous_sequences(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    first_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)
    sequence = _request_object(
        live_api_base_url,
        "GET",
        f"/v1/artifact-sequences/{first_result['source_sequence_id']}",
    )
    _request_object(
        live_api_base_url,
        "POST",
        "/v1/artifact-sequences",
        {
            "artifact_type": "source.page_image",
            "schema_version": 1,
            "item_refs": sequence["item_refs"],
            "index_key": sequence["index_key"],
            "metadata": {
                "project_id": first_result["project_id"],
                "source_id": first_result["source_id"],
                "page_count": first_result["source_page_count"],
            },
        },
    )

    completed = _run_script_failure(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_workflow.py",
            "--engine",
            "local.text",
            "--source-id",
            str(first_result["source_id"]),
        ],
    )

    assert completed.returncode != 0
    assert str(first_result["source_id"]) in completed.stderr
    assert "--sequence-id" in completed.stderr


def test_run_ocr_experiment_script_compares_engines_over_existing_sequence(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_experiment.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--engine",
            "local.text",
            "--engine",
            "local.text",
        ],
    )

    assert result["source_kind"] == "existing_sequence"
    assert result["source_id"] == source_result["source_id"]
    assert result["source_sequence_id"] == source_result["source_sequence_id"]
    assert result["workflow_template_id"] == "ocr-pages"
    assert result["source_page_count"] == 2
    assert result["experiment_status"] == "succeeded"
    assert result["variant_count"] == 2
    assert "summary.invocation_count" in result["metric_names"]
    variants = _list_field(result, "variants")
    assert [variant["engine"] for variant in variants] == [
        "local.text",
        "local.text",
    ]
    for variant in variants:
        assert variant["workflow_run_status"] == "succeeded"
        assert len(_list_field(variant, "processed_node_run_ids")) == 1
        assert variant["errors"] == []
        assert variant["artifact_counts"]["ocr.page_result"] == 2
        assert variant["artifact_counts"]["ocr.document_result"] == 1
        assert len(_list_field(variant, "output_sequence_ids")) == 1
        metric_values = _list_field(variant, "metric_values")
        assert any(
            metric["name"] == "summary.invocation_count"
            for metric in metric_values
        )
        assert [payload["text"] for payload in _list_field(variant, "ocr_payloads")] == [
            "first page text",
            "second page text",
        ]


def test_run_ocr_experiment_script_requires_two_engines(
    live_api_base_url: str,
) -> None:
    completed = _run_script_failure(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_experiment.py",
            "--engine",
            "local.text",
        ],
    )

    assert completed.returncode == 2
    assert "at least two --engine values are required" in completed.stderr


def test_run_ocr_experiment_script_rejects_wrong_sequence_type(
    live_api_base_url: str,
) -> None:
    artifact = _request_object(
        live_api_base_url,
        "POST",
        "/v1/artifacts",
        {
            "artifact_type": "debug.text",
            "schema_version": 1,
            "payload_ref": "artifact://debug/text.txt",
        },
    )
    sequence = _request_object(
        live_api_base_url,
        "POST",
        "/v1/artifact-sequences",
        {
            "artifact_type": "debug.text",
            "schema_version": 1,
            "item_refs": [
                {
                    "artifact_id": artifact["id"],
                    "artifact_type": "debug.text",
                    "schema_version": 1,
                }
            ],
        },
    )

    completed = _run_script_failure(
        live_api_base_url,
        [
            "scripts/platform/run_ocr_experiment.py",
            "--sequence-id",
            str(sequence["id"]),
            "--engine",
            "local.text",
            "--engine",
            "local.text",
        ],
    )

    assert completed.returncode != 0
    assert "source.page_image" in completed.stderr


def test_run_contextual_extraction_workflow_script_over_existing_sequence(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_workflow.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--policy-type",
            "accumulating",
            "--static-context-json",
            '{"corpus":"schematism","language":"Latin and Polish"}',
        ],
    )

    assert result["source_kind"] == "existing_sequence"
    assert result["source_id"] == source_result["source_id"]
    assert result["source_sequence_id"] == source_result["source_sequence_id"]
    assert result["workflow_template_id"] == "contextual-extraction"
    assert result["source_page_count"] == 2
    assert result["workflow_run_status"] == "succeeded"
    assert len(_list_field(result, "processed_node_run_ids")) == 8
    assert result["artifact_counts"]["context.bundle"] == 1
    assert result["artifact_counts"]["ocr.page_result"] == 2
    assert result["artifact_counts"]["ocr.document_result"] == 1
    assert result["artifact_counts"]["model.input"] == 2
    assert result["artifact_counts"]["model.response"] == 2
    assert result["artifact_counts"]["extraction.record_result"] == 2
    assert result["artifact_counts"]["extraction.document_result"] == 1
    assert result["artifact_counts"]["export.dataset"] == 1

    document_payload = _dict_field(result, "document_payload")
    assert document_payload["page_count"] == 2
    assert document_payload["records"] == [
        {"text": "first page text", "page_number": 1},
        {"text": "second page text", "page_number": 2},
    ]
    record_payloads = _list_field(result, "record_payloads")
    assert [record["record"] for record in record_payloads] == [
        {"text": "first page text", "page_number": 1},
        {"text": "second page text", "page_number": 2},
    ]
    model_inputs = _list_field(result, "model_input_payloads")
    first_context = _dict_field(model_inputs[0], "context")
    assert first_context["STATIC_CONTEXT"] == {
        "corpus": "schematism",
        "language": "Latin and Polish",
    }
    second_context = _dict_field(model_inputs[1], "context")
    assert second_context["PREVIOUS_RECORDS"] == [
        {"text": "first page text", "page_number": 1}
    ]
    static_context_payloads = _list_field(result, "static_context_payloads")
    assert _dict_field(static_context_payloads[0], "context") == {
        "corpus": "schematism",
        "language": "Latin and Polish",
    }
    export_payloads = _list_field(result, "export_dataset_payloads")
    export_payload = _dict_field(export_payloads[0], "metadata")
    assert export_payload["page_count"] == 2
    assert export_payload["policy_type"] == "accumulating"
    assert export_payloads[0]["records"] == document_payload["records"]
    trace_counts = _dict_field(result, "trace_counts")
    assert trace_counts == {"input_assembly": 8, "invocation": 10}


def test_run_contextual_extraction_workflow_script_returns_csv_export_text(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_workflow.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--export-format",
            "csv",
        ],
    )

    assert result["workflow_run_status"] == "succeeded"
    assert result["artifact_counts"]["export.dataset"] == 1
    assert _list_field(result, "export_dataset_payloads") == []
    assert _list_field(result, "export_dataset_texts") == [
        "text,page_number\nfirst page text,1\nsecond page text,2\n"
    ]


def test_run_contextual_extraction_experiment_script_varies_prompts(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_experiment.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--prompt-template-variant",
            "Variant A: {{ CURRENT_PAGE_TEXT }}",
            "--prompt-template-variant",
            "Variant B: {{ CURRENT_PAGE_TEXT }}",
        ],
    )

    assert result["source_kind"] == "existing_sequence"
    assert result["source_id"] == source_result["source_id"]
    assert result["source_sequence_id"] == source_result["source_sequence_id"]
    assert result["workflow_template_id"] == "contextual-extraction"
    assert result["experiment_status"] == "succeeded"
    assert result["variant_count"] == 2
    assert "summary.invocation_count" in result["metric_names"]
    variants = _list_field(result, "variants")
    assert [
        _dict_field(variant, "parameter_values")["template"]
        for variant in variants
    ] == [
        "Variant A: {{ CURRENT_PAGE_TEXT }}",
        "Variant B: {{ CURRENT_PAGE_TEXT }}",
    ]
    for variant in variants:
        assert variant["workflow_run_status"] == "succeeded"
        assert len(_list_field(variant, "processed_node_run_ids")) == 8
        assert variant["errors"] == []
        artifact_counts = _dict_field(variant, "artifact_counts")
        assert artifact_counts["ocr.page_result"] == 2
        assert artifact_counts["model.input"] == 2
        assert artifact_counts["extraction.document_result"] == 1
        assert artifact_counts["export.dataset"] == 1
        document_payload = _dict_field(variant, "document_payload")
        assert document_payload["records"] == [
            {"text": "first page text", "page_number": 1},
            {"text": "second page text", "page_number": 2},
        ]
        assert len(_list_field(variant, "model_input_payloads")) == 2
        assert _dict_field(variant, "trace_counts") == {
            "input_assembly": 8,
            "invocation": 10,
        }


def test_run_contextual_extraction_experiment_script_varies_export_format(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_experiment.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--export-format-variant",
            "json",
            "--export-format-variant",
            "csv",
        ],
    )

    variants = _list_field(result, "variants")
    assert result["experiment_status"] == "succeeded"
    assert result["variant_count"] == 2
    assert [
        _dict_field(variant, "parameter_values")["format"]
        for variant in variants
    ] == ["json", "csv"]
    json_variant = variants[0]
    csv_variant = variants[1]
    assert len(_list_field(json_variant, "export_dataset_payloads")) == 1
    assert _list_field(json_variant, "export_dataset_texts") == []
    assert _list_field(csv_variant, "export_dataset_payloads") == []
    assert _list_field(csv_variant, "export_dataset_texts") == [
        "text,page_number\nfirst page text,1\nsecond page text,2\n"
    ]


def test_run_contextual_extraction_experiment_script_requires_variant(
    live_api_base_url: str,
) -> None:
    completed = _run_script_failure(
        live_api_base_url,
        ["scripts/platform/run_contextual_extraction_experiment.py"],
    )

    assert completed.returncode == 2
    assert "at least one --*-variant option is required" in completed.stderr


def test_run_contextual_extraction_experiment_script_compares_ocr_streams(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_experiment.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--comparison-ocr-engine",
            "local.text",
            "--ocr-engine-variant",
            "local.text",
            "--comparison-ocr-engine-variant",
            "local.text",
            "--selected-candidate-variant",
            "candidate_a",
            "--selected-candidate-variant",
            "candidate_b",
            "--candidate-a-label",
            "Primary OCR",
            "--candidate-b-label",
            "Comparison OCR",
        ],
    )

    assert result["workflow_template_id"] == "ocr-compare-contextual-extraction"
    assert result["experiment_status"] == "succeeded"
    assert result["variant_count"] == 2
    variants = _list_field(result, "variants")
    assert [
        _dict_field(variant, "parameter_values")["selected_candidate"]
        for variant in variants
    ] == ["candidate_a", "candidate_b"]
    for variant in variants:
        parameter_values = _dict_field(variant, "parameter_values")
        assert parameter_values["ocr_a_engine"] == "local.text"
        assert parameter_values["ocr_b_engine"] == "local.text"
        assert variant["workflow_run_status"] == "succeeded"
        assert len(_list_field(variant, "processed_node_run_ids")) == 11
        assert variant["errors"] == []
        artifact_counts = _dict_field(variant, "artifact_counts")
        assert artifact_counts["ocr.page_result"] == 4
        assert artifact_counts["ocr.document_result"] == 2
        assert artifact_counts["ocr.comparison_result"] == 2
        assert artifact_counts["evaluation.metrics"] == 1
        assert artifact_counts["extraction.document_result"] == 1
        assert artifact_counts["export.dataset"] == 1
        comparison_payloads = _list_field(variant, "ocr_comparison_payloads")
        assert len(comparison_payloads) == 2
        assert comparison_payloads[0]["candidate_a_label"] == "Primary OCR"
        assert comparison_payloads[0]["candidate_b_label"] == "Comparison OCR"
        document_payload = _dict_field(variant, "document_payload")
        assert document_payload["records"] == [
            {"text": "first page text", "page_number": 1},
            {"text": "second page text", "page_number": 2},
        ]


def test_run_contextual_extraction_script_compares_ocr_before_extraction(
    live_api_base_url: str,
    tmp_path: Path,
) -> None:
    source_result = _run_uploaded_text_pages(live_api_base_url, tmp_path)

    result = _run_script(
        live_api_base_url,
        [
            "scripts/platform/run_contextual_extraction_workflow.py",
            "--sequence-id",
            str(source_result["source_sequence_id"]),
            "--ocr-engine",
            "local.text",
            "--comparison-ocr-engine",
            "local.text",
            "--selected-candidate",
            "candidate_b",
            "--candidate-a-label",
            "Local A",
            "--candidate-b-label",
            "Local B",
        ],
    )

    assert result["workflow_template_id"] == "ocr-compare-contextual-extraction"
    assert result["workflow_run_status"] == "succeeded"
    assert len(_list_field(result, "processed_node_run_ids")) == 11
    assert result["artifact_counts"]["ocr.page_result"] == 4
    assert result["artifact_counts"]["ocr.document_result"] == 2
    assert result["artifact_counts"]["ocr.comparison_result"] == 2
    assert result["artifact_counts"]["evaluation.metrics"] == 1
    assert result["artifact_counts"]["extraction.record_result"] == 2
    assert result["artifact_counts"]["extraction.document_result"] == 1
    assert result["artifact_counts"]["export.dataset"] == 1
    document_payload = _dict_field(result, "document_payload")
    assert document_payload["records"] == [
        {"text": "first page text", "page_number": 1},
        {"text": "second page text", "page_number": 2},
    ]
    export_payloads = _list_field(result, "export_dataset_payloads")
    assert export_payloads[0]["records"] == document_payload["records"]


def _run_uploaded_text_pages(base_url: str, tmp_path: Path) -> dict[str, object]:
    first_page = tmp_path / f"page-1-{time.monotonic_ns()}.png"
    second_page = tmp_path / f"page-2-{time.monotonic_ns()}.png"
    first_page.write_text("first page text", encoding="utf-8")
    second_page.write_text("second page text", encoding="utf-8")
    return _run_script(
        base_url,
        [
            "scripts/platform/run_ocr_workflow.py",
            "--engine",
            "local.text",
            "--image",
            str(first_page),
            "--image",
            str(second_page),
        ],
    )


def _run_script(base_url: str, args: list[str]) -> dict[str, object]:
    env = {
        **os.environ,
        "NOTARIUS_API_BASE_URL": base_url,
    }
    completed = subprocess.run(
        [sys.executable, *args],
        check=True,
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, dict):
        raise AssertionError("Script stdout did not contain a JSON object")
    return parsed


def _run_script_failure(
    base_url: str,
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "NOTARIUS_API_BASE_URL": base_url,
    }
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        cwd=Path(__file__).parents[3],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _request_object(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{base_url}{path}", data=data, headers=headers, method=method)
    with urlopen(request, timeout=30) as response:
        parsed = json.loads(response.read().decode("utf-8"))
    if not isinstance(parsed, dict):
        raise AssertionError(f"{method} {path} did not return a JSON object")
    return parsed


def _list_field(value: dict[str, object], field_name: str) -> list[object]:
    field_value = value[field_name]
    if not isinstance(field_value, list):
        raise AssertionError(f"{field_name} is not a list")
    return field_value


def _dict_field(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError("Value is not a dict")
    field_value = value[field_name]
    if not isinstance(field_value, dict):
        raise AssertionError(f"{field_name} is not a dict")
    return field_value


def _pdf_bytes(pages: list[str]) -> bytes:
    document = fitz.open()
    try:
        for text in pages:
            page = document.new_page()
            page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


class _FakeMistralOcrHandler(BaseHTTPRequestHandler):
    request_payloads: list[dict[str, object]] = []

    def do_POST(self) -> None:
        if self.path != "/v1/ocr":
            self.send_error(404)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        payload = json.loads(raw_body.decode("utf-8"))
        if not isinstance(payload, dict):
            self.send_error(400)
            return
        self.request_payloads.append(payload)
        page_number = len(self.request_payloads)
        response = {
            "pages": [
                {
                    "markdown": f"fake mistral page {page_number}",
                    "blocks": [
                        {
                            "type": "text",
                            "text": f"fake mistral page {page_number}",
                        }
                    ],
                    "dimensions": {"confidence": 0.97},
                }
            ]
        }
        response_bytes = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format: str, *args: object) -> None:
        return


def _start_fake_mistral_ocr_server() -> tuple[str, ThreadingHTTPServer, threading.Thread]:
    _FakeMistralOcrHandler.request_payloads = []
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _FakeMistralOcrHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}", server, thread


def _wait_for_api(base_url: str) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health/live", timeout=0.5) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"API server did not become ready at {base_url}")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
