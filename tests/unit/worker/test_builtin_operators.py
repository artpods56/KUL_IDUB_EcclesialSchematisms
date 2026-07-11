from collections.abc import Callable, Mapping
from io import BytesIO
import json
from uuid import UUID, uuid4

import httpx
import pytest
from PIL import Image
from pydantic import BaseModel

from notarius_core.application.operators import (
    CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
    CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    CONTEXT_STATIC_DEFINE_OPERATOR_ID,
    CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
    DEBUG_EMIT_TEXT_OPERATOR_ID,
    DEBUG_EMIT_TEXT_OPERATOR_VERSION,
    EXPORT_DATASET_OPERATOR_ID,
    EXPORT_DATASET_OPERATOR_VERSION,
    EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
    EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
    INPUT_POLICY_DEFINE_OPERATOR_ID,
    INPUT_POLICY_DEFINE_OPERATOR_VERSION,
    MODEL_BINDING_DEFINE_OPERATOR_ID,
    MODEL_BINDING_DEFINE_OPERATOR_VERSION,
    OCR_COLLECT_PAGES_OPERATOR_ID,
    OCR_COLLECT_PAGES_OPERATOR_VERSION,
    OCR_COMPARE_PAGES_OPERATOR_ID,
    OCR_COMPARE_PAGES_OPERATOR_VERSION,
    OCR_EXTRACT_PAGE_OPERATOR_ID,
    OCR_EXTRACT_PAGE_OPERATOR_VERSION,
    OCR_EXTRACT_PAGES_OPERATOR_ID,
    OCR_EXTRACT_PAGES_OPERATOR_VERSION,
    OCR_SELECT_PAGES_OPERATOR_ID,
    OCR_SELECT_PAGES_OPERATOR_VERSION,
    PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
    PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
    SCHEMA_VALIDATION_OPERATOR_ID,
    SCHEMA_VALIDATION_OPERATOR_VERSION,
)
from notarius_core.domain.models import Artifact, ArtifactSequence, NodeRun
from notarius_core.ports.unit_of_work import StudioUnitOfWorkPort
from notarius_storage import (
    LocalArtifactPayloadStorage,
    SaveArtifactPayloadCommand,
    artifact_payload_ref,
    parse_artifact_payload_ref,
)
from notarius_worker import operators as worker_operators
from notarius_worker import streaming
from notarius_worker.node_execution import (
    ArtifactSequenceInput,
    NodeExecutionRequest,
    NodeRunExecutionError,
    NodeRunHandler,
)
from notarius_worker.operators import (
    ContextualStructuredExtractionHandler,
    ContextualExtractionPageInput,
    ContextBundlePayload,
    EmitTextHandler,
    EvaluationMetricsPayload,
    ExportDatasetHandler,
    ExportDatasetPayload,
    ExtractionSchemaDefineHandler,
    ExtractionDocumentResultPayload,
    ExtractionRecordResultPayload,
    ExtractionSchemaPayload,
    InputPolicyDefineHandler,
    InputPolicyPayload,
    MistralOcrEngine,
    ModelBindingDefineHandler,
    ModelBindingPayload,
    ModelInputPayload,
    OcrComparePagesHandler,
    OcrCollectPagesHandler,
    OcrComparisonPagePayload,
    OcrDocumentResultPayload,
    OcrPageInput,
    OcrPageEngineError,
    OcrPageResultPayload,
    OcrRequestTracePayload,
    OcrResponseTracePayload,
    OcrExtractPageHandler,
    OcrExtractPagesHandler,
    OcrSelectPagesHandler,
    OpenAICompatibleStructuredExtractionEngine,
    PromptTemplateDefineHandler,
    PromptTemplatePayload,
    SchemaValidationHandler,
    StaticContextDefineHandler,
    StructuredExtractionEngineError,
    TesseractOcrEngine,
    ValidationResultPayload,
    builtin_node_handlers,
)


class RetryableOcrEngine:
    engine_id = "provider.retryable"

    def extract_page(self, page: OcrPageInput) -> OcrPageResultPayload:
        raise OcrPageEngineError("provider rate limited", retryable=True)


@pytest.mark.asyncio
async def test_emit_text_handler_emits_debug_text_artifact() -> None:
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="emit",
        operator_id=DEBUG_EMIT_TEXT_OPERATOR_ID,
        operator_version=DEBUG_EMIT_TEXT_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "text": "sample output",
                "payload_ref": "memory://custom/output.txt",
            },
        },
    )

    result = await EmitTextHandler().execute(
        NodeExecutionRequest(node_run=node_run, input_artifacts={})
    )

    artifact = result.artifacts[0]
    assert artifact.artifact_type == "debug.text"
    assert artifact.schema_version == 1
    assert artifact.workflow_run_id == node_run.workflow_run_id
    assert artifact.producer_node_run_id == node_run.id
    assert artifact.producer_operator_id == DEBUG_EMIT_TEXT_OPERATOR_ID
    assert artifact.payload_ref == "memory://custom/output.txt"
    assert artifact.metadata == {"text": "sample output"}
    assert result.output_artifact_refs == {"text": artifact.ref()}
    assert result.invocation_traces[0].invocation_type == DEBUG_EMIT_TEXT_OPERATOR_ID
    assert result.invocation_traces[0].runtime == {"text_length": 13}


@pytest.mark.asyncio
async def test_emit_text_handler_rejects_missing_text_config() -> None:
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="emit",
        operator_id=DEBUG_EMIT_TEXT_OPERATOR_ID,
        operator_version=DEBUG_EMIT_TEXT_OPERATOR_VERSION,
        metadata={"workflow_node_config": {}},
    )

    with pytest.raises(NodeRunExecutionError, match="text to be a string"):
        await EmitTextHandler().execute(
            NodeExecutionRequest(node_run=node_run, input_artifacts={})
        )


def test_builtin_node_handlers_registers_emit_text_handler() -> None:
    handlers = builtin_node_handlers()

    assert (
        DEBUG_EMIT_TEXT_OPERATOR_ID,
        DEBUG_EMIT_TEXT_OPERATOR_VERSION,
    ) in handlers
    assert (
        OCR_SELECT_PAGES_OPERATOR_ID,
        OCR_SELECT_PAGES_OPERATOR_VERSION,
    ) in handlers


def test_builtin_node_handlers_registers_ocr_handler_when_storage_is_supplied(
    tmp_path,
) -> None:
    handlers = builtin_node_handlers(LocalArtifactPayloadStorage(tmp_path))

    assert (
        CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
        CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    ) in handlers
    assert (
        CONTEXT_STATIC_DEFINE_OPERATOR_ID,
        CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
    ) in handlers
    assert (
        EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
        EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
    ) in handlers
    assert (
        EXPORT_DATASET_OPERATOR_ID,
        EXPORT_DATASET_OPERATOR_VERSION,
    ) in handlers
    assert (
        INPUT_POLICY_DEFINE_OPERATOR_ID,
        INPUT_POLICY_DEFINE_OPERATOR_VERSION,
    ) in handlers
    assert (
        MODEL_BINDING_DEFINE_OPERATOR_ID,
        MODEL_BINDING_DEFINE_OPERATOR_VERSION,
    ) in handlers
    assert (
        OCR_COMPARE_PAGES_OPERATOR_ID,
        OCR_COMPARE_PAGES_OPERATOR_VERSION,
    ) in handlers
    assert (
        OCR_COLLECT_PAGES_OPERATOR_ID,
        OCR_COLLECT_PAGES_OPERATOR_VERSION,
    ) in handlers
    assert (
        OCR_EXTRACT_PAGE_OPERATOR_ID,
        OCR_EXTRACT_PAGE_OPERATOR_VERSION,
    ) in handlers
    assert (
        OCR_EXTRACT_PAGES_OPERATOR_ID,
        OCR_EXTRACT_PAGES_OPERATOR_VERSION,
    ) in handlers
    assert (
        OCR_SELECT_PAGES_OPERATOR_ID,
        OCR_SELECT_PAGES_OPERATOR_VERSION,
    ) in handlers
    assert (
        PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
        PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
    ) in handlers
    assert (
        SCHEMA_VALIDATION_OPERATOR_ID,
        SCHEMA_VALIDATION_OPERATOR_VERSION,
    ) in handlers


@pytest.mark.asyncio
async def test_prompt_template_define_handler_emits_artifact(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="prompt",
        operator_id=PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
        operator_version=PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "name": "Page extraction prompt",
                "template": "Extract records from {{ CURRENT_PAGE_TEXT }}",
                "template_format": "jinja2",
                "variables": ["CURRENT_PAGE_TEXT"],
                "description": "Prompt for page-level extraction",
            }
        },
    )

    result = await PromptTemplateDefineHandler(storage).execute(
        NodeExecutionRequest(node_run=node_run, input_artifacts={})
    )

    artifact = result.artifacts[0]
    payload = PromptTemplatePayload.model_validate_json(
        _load_payload_bytes(storage, artifact)
    )
    assert result.output_artifact_refs == {"template": artifact.ref()}
    assert artifact.artifact_type == "prompt.template"
    assert artifact.metadata["name"] == "Page extraction prompt"
    assert artifact.metadata["template_format"] == "jinja2"
    assert artifact.metadata["variable_count"] == 1
    assert payload.template == "Extract records from {{ CURRENT_PAGE_TEXT }}"
    assert payload.variables == ["CURRENT_PAGE_TEXT"]
    assert result.input_assembly_traces[0].selected_inputs == {}
    assert result.invocation_traces[0].invocation_type == (
        PROMPT_TEMPLATE_DEFINE_OPERATOR_ID
    )
    assert result.invocation_traces[0].output_artifact_refs == [artifact.ref()]


@pytest.mark.asyncio
async def test_extraction_schema_define_handler_emits_artifact(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    json_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="schema",
        operator_id=EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
        operator_version=EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "name": "Person schema",
                "json_schema": json_schema,
            }
        },
    )

    result = await ExtractionSchemaDefineHandler(storage).execute(
        NodeExecutionRequest(node_run=node_run, input_artifacts={})
    )

    artifact = result.artifacts[0]
    payload = ExtractionSchemaPayload.model_validate_json(
        _load_payload_bytes(storage, artifact)
    )
    assert result.output_artifact_refs == {"schema": artifact.ref()}
    assert artifact.artifact_type == "extraction.schema"
    assert artifact.metadata["name"] == "Person schema"
    assert artifact.metadata["schema_format"] == "json_schema"
    assert payload.json_schema == json_schema
    assert result.invocation_traces[0].runtime == {
        "top_level_keys": ["properties", "required", "type"]
    }


@pytest.mark.asyncio
async def test_model_binding_define_handler_emits_artifact(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="model",
        operator_id=MODEL_BINDING_DEFINE_OPERATOR_ID,
        operator_version=MODEL_BINDING_DEFINE_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "provider": "openai-compatible",
                "model": "vision-model",
                "parameters": {"temperature": 0, "max_output_tokens": 2048},
                "capabilities": ["vision", "structured_output"],
                "credential_ref": "secret://providers/openai-compatible",
            }
        },
    )

    result = await ModelBindingDefineHandler(storage).execute(
        NodeExecutionRequest(node_run=node_run, input_artifacts={})
    )

    artifact = result.artifacts[0]
    payload = ModelBindingPayload.model_validate_json(
        _load_payload_bytes(storage, artifact)
    )
    assert result.output_artifact_refs == {"binding": artifact.ref()}
    assert artifact.artifact_type == "model.binding"
    assert artifact.metadata["provider"] == "openai-compatible"
    assert artifact.metadata["model"] == "vision-model"
    assert artifact.metadata["has_credential_ref"] is True
    assert payload.parameters == {"temperature": 0, "max_output_tokens": 2048}
    assert payload.credential_ref == "secret://providers/openai-compatible"
    assert result.invocation_traces[0].runtime == {
        "model": "vision-model",
        "capabilities": ["vision", "structured_output"],
        "parameter_keys": ["max_output_tokens", "temperature"],
    }


@pytest.mark.asyncio
async def test_model_binding_define_handler_rejects_sensitive_config(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="model",
        operator_id=MODEL_BINDING_DEFINE_OPERATOR_ID,
        operator_version=MODEL_BINDING_DEFINE_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "provider": "openai-compatible",
                "model": "vision-model",
                "parameters": {"api_key": "secret"},
            }
        },
    )

    with pytest.raises(NodeRunExecutionError, match="credential_ref"):
        await ModelBindingDefineHandler(storage).execute(
            NodeExecutionRequest(node_run=node_run, input_artifacts={})
        )


@pytest.mark.asyncio
async def test_input_policy_define_handler_emits_artifact(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="policy",
        operator_id=INPUT_POLICY_DEFINE_OPERATOR_ID,
        operator_version=INPUT_POLICY_DEFINE_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "name": "Sliding window with image pruning",
                "policy_type": "sliding_window",
                "settings": {"window_size": 3, "strip_images": True},
                "applies_to": ["pages", "text"],
            }
        },
    )

    result = await InputPolicyDefineHandler(storage).execute(
        NodeExecutionRequest(node_run=node_run, input_artifacts={})
    )

    artifact = result.artifacts[0]
    payload = InputPolicyPayload.model_validate_json(
        _load_payload_bytes(storage, artifact)
    )
    assert result.output_artifact_refs == {"policy": artifact.ref()}
    assert artifact.artifact_type == "input.policy"
    assert artifact.metadata["policy_type"] == "sliding_window"
    assert payload.settings == {"window_size": 3, "strip_images": True}
    assert payload.applies_to == ["pages", "text"]
    assert result.input_assembly_traces[0].policies == {
        "policy_type": "sliding_window"
    }
    assert result.invocation_traces[0].runtime == {
        "setting_keys": ["strip_images", "window_size"],
        "applies_to": ["pages", "text"],
    }


@pytest.mark.asyncio
async def test_static_context_define_handler_emits_context_bundle(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="context",
        operator_id=CONTEXT_STATIC_DEFINE_OPERATOR_ID,
        operator_version=CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "name": "Research notes",
                "context": {
                    "corpus": "schematism",
                    "language": "Latin and Polish",
                },
                "applies_to": ["text"],
            }
        },
    )

    result = await StaticContextDefineHandler(storage).execute(
        NodeExecutionRequest(node_run=node_run, input_artifacts={})
    )

    artifact = result.artifacts[0]
    payload = ContextBundlePayload.model_validate_json(
        _load_payload_bytes(storage, artifact)
    )
    assert result.output_artifact_refs == {"context": artifact.ref()}
    assert artifact.artifact_type == "context.bundle"
    assert artifact.metadata["name"] == "Research notes"
    assert artifact.metadata["context_keys"] == ["corpus", "language"]
    assert payload.context == {
        "corpus": "schematism",
        "language": "Latin and Polish",
    }
    assert result.input_assembly_traces[0].policies == {"applies_to": ["text"]}
    assert result.invocation_traces[0].runtime == {
        "context_keys": ["corpus", "language"]
    }


@pytest.mark.asyncio
async def test_export_dataset_handler_emits_json_export_artifact(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    document_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="extraction.document_result",
        bucket="extraction-document-results",
        key="document-result.json",
        payload=ExtractionDocumentResultPayload(
            page_count=2,
            record_count=2,
            records=[
                {"name": "Alpha", "page_number": 1},
                {"name": "Beta", "page_number": 2},
            ],
            validation_error_count=0,
            page_result_artifact_ids=[str(uuid4()), str(uuid4())],
            model_input_sequence_id=str(uuid4()),
            model_response_sequence_id=str(uuid4()),
            provider="local",
            model="echo",
            policy_type="accumulating",
        ),
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="export",
        operator_id=EXPORT_DATASET_OPERATOR_ID,
        operator_version=EXPORT_DATASET_OPERATOR_VERSION,
        metadata={"workflow_node_config": {"format": "json"}},
    )

    result = await ExportDatasetHandler(storage).execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={"document": document_artifact},
        )
    )

    artifact = result.artifacts[0]
    payload = ExportDatasetPayload.model_validate_json(
        _load_payload_bytes(storage, artifact)
    )
    assert result.output_artifact_refs == {"dataset": artifact.ref()}
    assert artifact.artifact_type == "export.dataset"
    assert artifact.input_artifact_ids == [document_artifact.id]
    assert artifact.metadata["format"] == "json"
    assert artifact.metadata["filename"] == "dataset.json"
    assert artifact.metadata["content_type"] == "application/json"
    assert artifact.metadata["record_count"] == 2
    assert payload.records == [
        {"name": "Alpha", "page_number": 1},
        {"name": "Beta", "page_number": 2},
    ]
    assert payload.metadata["provider"] == "local"
    assert result.input_assembly_traces[0].selected_inputs == {
        "document": document_artifact.ref()
    }
    assert result.invocation_traces[0].runtime == {
        "format": "json",
        "record_count": 2,
    }


@pytest.mark.asyncio
async def test_export_dataset_handler_emits_csv_export_artifact(tmp_path) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    document_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="extraction.document_result",
        bucket="extraction-document-results",
        key="document-result.json",
        payload=ExtractionDocumentResultPayload(
            page_count=1,
            record_count=1,
            records=[
                {
                    "name": "Alpha",
                    "page_number": 1,
                    "aliases": ["A", "Alpha Parish"],
                }
            ],
            validation_error_count=0,
            page_result_artifact_ids=[str(uuid4())],
            model_input_sequence_id=str(uuid4()),
            model_response_sequence_id=str(uuid4()),
            provider="local",
            model="echo",
            policy_type="stateless",
        ),
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="export",
        operator_id=EXPORT_DATASET_OPERATOR_ID,
        operator_version=EXPORT_DATASET_OPERATOR_VERSION,
        metadata={"workflow_node_config": {"format": "csv"}},
    )

    result = await ExportDatasetHandler(storage).execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={"document": document_artifact},
        )
    )

    artifact = result.artifacts[0]
    csv_payload = _load_payload_bytes(storage, artifact).decode("utf-8")
    assert artifact.metadata["format"] == "csv"
    assert artifact.metadata["filename"] == "dataset.csv"
    assert artifact.metadata["content_type"] == "text/csv"
    assert csv_payload == (
        "name,page_number,aliases\n"
        'Alpha,1,"[""A"", ""Alpha Parish""]"\n'
    )


@pytest.mark.asyncio
async def test_schema_validation_handler_emits_result_and_metrics_artifacts(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    schema_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="extraction.schema",
        bucket="schemas",
        key="person-schema.json",
        payload=ExtractionSchemaPayload(
            name="Person",
            json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
                "additionalProperties": False,
            },
        ),
    )
    document_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="extraction.document_result",
        bucket="documents",
        key="document-result.json",
        payload=ExtractionDocumentResultPayload(
            page_count=2,
            record_count=2,
            records=[
                {"name": "Jan", "age": 42},
                {"name": ""},
            ],
            validation_error_count=0,
            page_result_artifact_ids=[],
            model_input_sequence_id=str(uuid4()),
            model_response_sequence_id=str(uuid4()),
            provider="local",
            model="echo",
            policy_type="stateless",
        ),
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="validate",
        operator_id=SCHEMA_VALIDATION_OPERATOR_ID,
        operator_version=SCHEMA_VALIDATION_OPERATOR_VERSION,
        metadata={"workflow_node_config": {}},
    )

    result = await SchemaValidationHandler(storage).execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={
                "document": document_artifact,
                "schema": schema_artifact,
            },
        )
    )

    validation_artifact = result.artifacts[0]
    metrics_artifact = result.artifacts[1]
    validation_payload = ValidationResultPayload.model_validate_json(
        _load_payload_bytes(storage, validation_artifact)
    )
    metrics_payload = EvaluationMetricsPayload.model_validate_json(
        _load_payload_bytes(storage, metrics_artifact)
    )

    assert result.output_artifact_refs == {
        "validation": validation_artifact.ref(),
        "metrics": metrics_artifact.ref(),
    }
    assert validation_artifact.artifact_type == "validation.result"
    assert validation_artifact.input_artifact_ids == [
        document_artifact.id,
        schema_artifact.id,
    ]
    assert validation_payload.valid is False
    assert validation_payload.record_count == 2
    assert validation_payload.valid_record_count == 1
    assert validation_payload.invalid_record_count == 1
    assert validation_payload.error_count == 2
    assert {error["record_index"] for error in validation_payload.errors} == {2}
    assert metrics_artifact.artifact_type == "evaluation.metrics"
    assert metrics_payload.metric_family == "schema_validation"
    assert metrics_payload.metrics == {
        "record_count": 2,
        "valid_record_count": 1,
        "invalid_record_count": 1,
        "error_count": 2,
        "valid": False,
    }
    assert result.input_assembly_traces[0].selected_inputs == {
        "document": document_artifact.ref(),
        "schema": schema_artifact.ref(),
    }
    assert result.invocation_traces[0].runtime == {
        "record_count": 2,
        "error_count": 2,
        "valid": False,
    }


@pytest.mark.asyncio
async def test_contextual_structured_extraction_handler_emits_auditable_sequences(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    prompt_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="prompt.template",
        bucket="prompt-templates",
        key="prompt.json",
        payload=PromptTemplatePayload(
            name="Page prompt",
            template=(
                "Page {{ CURRENT_PAGE_NUMBER }}: {{ CURRENT_PAGE_TEXT }} "
                "previous={{ PREVIOUS_RECORD.text if PREVIOUS_RECORD else 'none' }}"
            ),
            variables=["CURRENT_PAGE_TEXT", "PREVIOUS_RECORD"],
        ),
    )
    schema_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="extraction.schema",
        bucket="extraction-schemas",
        key="schema.json",
        payload=ExtractionSchemaPayload(
            name="Text schema",
            json_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "page_number": {"type": "integer"},
                },
                "required": ["text", "page_number"],
                "additionalProperties": False,
            },
        ),
    )
    binding_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="model.binding",
        bucket="model-bindings",
        key="binding.json",
        payload=ModelBindingPayload(
            provider="local",
            model="echo",
            capabilities=["structured_output"],
        ),
    )
    policy_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="input.policy",
        bucket="input-policies",
        key="policy.json",
        payload=InputPolicyPayload(
            name="Accumulate previous records",
            policy_type="accumulating",
            applies_to=["text"],
        ),
    )
    context_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="context.bundle",
        bucket="context-bundles",
        key="context.json",
        payload=ContextBundlePayload(
            name="Static research notes",
            context={"corpus": "schematism", "language": "Latin and Polish"},
            applies_to=["text"],
        ),
    )
    first_ocr = _save_ocr_page_result(
        storage,
        workflow_run_id=workflow_run_id,
        key="ocr/page-1.json",
        page_number=1,
        engine="local.text",
        text="Alpha page text",
    )
    second_ocr = _save_ocr_page_result(
        storage,
        workflow_run_id=workflow_run_id,
        key="ocr/page-2.json",
        page_number=2,
        engine="local.text",
        text="Beta page text",
    )
    first_page = _page_image_artifact(workflow_run_id=workflow_run_id, page_number=1)
    second_page = _page_image_artifact(workflow_run_id=workflow_run_id, page_number=2)
    text_sequence = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[first_ocr.ref(), second_ocr.ref()],
        index_key="page_number",
    )
    page_sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=[first_page.ref(), second_page.ref()],
        index_key="page_number",
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="extract",
        operator_id=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
        operator_version=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    )

    result = await ContextualStructuredExtractionHandler(storage).execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={
                "text": ArtifactSequenceInput(
                    sequence=text_sequence,
                    artifacts=[first_ocr, second_ocr],
                ),
                "pages": ArtifactSequenceInput(
                    sequence=page_sequence,
                    artifacts=[first_page, second_page],
                ),
                "schema": schema_artifact,
                "template": prompt_artifact,
                "binding": binding_artifact,
                "policy": policy_artifact,
                "context": context_artifact,
            },
        )
    )

    sequences_by_type = {
        sequence.artifact_type: sequence for sequence in result.artifact_sequences
    }
    document_artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "extraction.document_result"
    )
    model_input_artifacts = [
        artifact for artifact in result.artifacts if artifact.artifact_type == "model.input"
    ]
    model_response_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "model.response"
    ]
    page_result_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "extraction.record_result"
    ]
    first_model_input = ModelInputPayload.model_validate_json(
        _load_payload_bytes(storage, model_input_artifacts[0])
    )
    second_model_input = ModelInputPayload.model_validate_json(
        _load_payload_bytes(storage, model_input_artifacts[1])
    )
    first_record = ExtractionRecordResultPayload.model_validate_json(
        _load_payload_bytes(storage, page_result_artifacts[0])
    )
    document = ExtractionDocumentResultPayload.model_validate_json(
        _load_payload_bytes(storage, document_artifact)
    )

    assert result.output_artifact_refs == {
        "page_results": sequences_by_type["extraction.record_result"].ref(),
        "document_result": document_artifact.ref(),
        "model_inputs": sequences_by_type["model.input"].ref(),
        "model_responses": sequences_by_type["model.response"].ref(),
    }
    assert [len(sequence.item_refs) for sequence in result.artifact_sequences] == [
        2,
        2,
        2,
    ]
    assert len(model_input_artifacts) == 2
    assert len(model_response_artifacts) == 2
    assert len(page_result_artifacts) == 2
    assert first_model_input.rendered_prompt == "Page 1: Alpha page text previous=none"
    assert second_model_input.context["PREVIOUS_RECORD"] == {
        "text": "Alpha page text",
        "page_number": 1,
    }
    assert first_model_input.context["STATIC_CONTEXT"] == {
        "corpus": "schematism",
        "language": "Latin and Polish",
    }
    assert first_model_input.context_bundle_artifact_id == str(context_artifact.id)
    assert first_record.record == {"text": "Alpha page text", "page_number": 1}
    assert first_record.validation_errors == []
    assert document.page_count == 2
    assert document.records[1] == {"text": "Beta page text", "page_number": 2}
    assert result.input_assembly_traces[0].selected_inputs["schema"] == (
        schema_artifact.ref()
    )
    assert result.input_assembly_traces[0].selected_inputs["context"] == (
        context_artifact.ref()
    )
    assert result.input_assembly_traces[0].selected_inputs["text"] == [
        first_ocr.ref(),
        second_ocr.ref(),
    ]
    assert len(result.invocation_traces) == 2
    assert result.invocation_traces[0].request_ref == model_input_artifacts[0].payload_ref
    assert result.invocation_traces[0].response_ref == (
        model_response_artifacts[0].payload_ref
    )
    assert result.invocation_traces[0].output_artifact_refs == [
        model_response_artifacts[0].ref(),
        page_result_artifacts[0].ref(),
    ]


@pytest.mark.asyncio
async def test_contextual_structured_extraction_handler_preserves_provider_retry(
    tmp_path,
) -> None:
    class RetryableStructuredEngine:
        engine_id = "openai-compatible"

        def extract_page(
            self,
            page: ContextualExtractionPageInput,
            binding: ModelBindingPayload,
        ) -> dict[str, object]:
            raise StructuredExtractionEngineError("provider busy", retryable=True)

    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    prompt_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="prompt.template",
        bucket="prompt-templates",
        key="prompt.json",
        payload=PromptTemplatePayload(
            name="Page prompt",
            template="Page {{ CURRENT_PAGE_NUMBER }}: {{ CURRENT_PAGE_TEXT }}",
        ),
    )
    schema_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="extraction.schema",
        bucket="extraction-schemas",
        key="schema.json",
        payload=ExtractionSchemaPayload(
            name="Text schema",
            json_schema={"type": "object"},
        ),
    )
    binding_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="model.binding",
        bucket="model-bindings",
        key="binding.json",
        payload=ModelBindingPayload(
            provider="openai-compatible",
            model="test-model",
        ),
    )
    policy_artifact = _save_payload_artifact(
        storage,
        workflow_run_id=workflow_run_id,
        artifact_type="input.policy",
        bucket="input-policies",
        key="policy.json",
        payload=InputPolicyPayload(
            name="Stateless",
            policy_type="stateless",
        ),
    )
    ocr_artifact = _save_ocr_page_result(
        storage,
        workflow_run_id=workflow_run_id,
        key="ocr/page-1.json",
        page_number=1,
        engine="local.text",
        text="Alpha page text",
    )
    text_sequence = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[ocr_artifact.ref()],
        index_key="page_number",
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="extract",
        operator_id=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
        operator_version=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    )

    with pytest.raises(NodeRunExecutionError, match="provider busy") as exc:
        await ContextualStructuredExtractionHandler(
            storage,
            engines={"openai-compatible": RetryableStructuredEngine()},
        ).execute(
            NodeExecutionRequest(
                node_run=node_run,
                input_artifacts={
                    "text": ArtifactSequenceInput(
                        sequence=text_sequence,
                        artifacts=[ocr_artifact],
                    ),
                    "schema": schema_artifact,
                    "template": prompt_artifact,
                    "binding": binding_artifact,
                    "policy": policy_artifact,
                },
            )
        )

    assert exc.value.retryable is True
    assert "openai-compatible" in str(exc.value)
    assert "test-model" in str(exc.value)
    assert "page 1" in str(exc.value)


@pytest.mark.asyncio
async def test_contextual_structured_extraction_handler_rejects_mismatched_pages(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    first_ocr = _ocr_artifact(
        workflow_run_id=workflow_run_id,
        page_number=1,
        engine="local.text",
    )
    second_ocr = _ocr_artifact(
        workflow_run_id=workflow_run_id,
        page_number=2,
        engine="local.text",
    )
    page = _page_image_artifact(workflow_run_id=workflow_run_id, page_number=1)
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="extract",
        operator_id=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
        operator_version=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
    )

    with pytest.raises(NodeRunExecutionError, match="pages to match text count"):
        await ContextualStructuredExtractionHandler(storage).execute(
            NodeExecutionRequest(
                node_run=node_run,
                input_artifacts={
                    "text": ArtifactSequenceInput(
                        sequence=ArtifactSequence(
                            artifact_type="ocr.page_result",
                            schema_version=1,
                            item_refs=[first_ocr.ref(), second_ocr.ref()],
                        ),
                        artifacts=[first_ocr, second_ocr],
                    ),
                    "pages": ArtifactSequenceInput(
                        sequence=ArtifactSequence(
                            artifact_type="source.page_image",
                            schema_version=1,
                            item_refs=[page.ref()],
                        ),
                        artifacts=[page],
                    ),
                },
            )
        )


@pytest.mark.asyncio
async def test_emit_text_handler_rejects_empty_payload_ref() -> None:
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="emit",
        operator_id=DEBUG_EMIT_TEXT_OPERATOR_ID,
        operator_version=DEBUG_EMIT_TEXT_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "text": "sample output",
                "payload_ref": "",
            },
        },
    )

    with pytest.raises(NodeRunExecutionError, match="payload_ref to be non-empty"):
        await EmitTextHandler().execute(
            NodeExecutionRequest(node_run=node_run, input_artifacts={})
        )


@pytest.mark.asyncio
async def test_ocr_extract_pages_handler_emits_ordered_page_result_sequence(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    first_payload = storage.save(
        SaveArtifactPayloadCommand(
            bucket="source-page-images",
            key="pages/page-1.png",
            payload=b"first page text",
        )
    )
    second_payload = storage.save(
        SaveArtifactPayloadCommand(
            bucket="source-page-images",
            key="pages/page-2.png",
            payload=b"second page text",
        )
    )
    workflow_run_id = uuid4()
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="ocr",
        operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
        operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "engine": "local.text",
                "language_hints": ["pl", "la"],
                "engine_config": {
                    "api_key": "do-not-store",
                    "timeout_seconds": 12,
                },
            }
        },
    )
    first_page = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref=artifact_payload_ref(
            bucket=first_payload.bucket,
            key=first_payload.key,
        ),
        content_hash=first_payload.sha256,
        metadata={"page_number": 10},
    )
    second_page = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref=artifact_payload_ref(
            bucket=second_payload.bucket,
            key=second_payload.key,
        ),
        content_hash=second_payload.sha256,
        metadata={"page_number": 11},
    )
    sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=[first_page.ref(), second_page.ref()],
        index_key="page_number",
    )

    result = await OcrExtractPagesHandler(storage).execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={
                "pages": ArtifactSequenceInput(
                    sequence=sequence,
                    artifacts=[first_page, second_page],
                )
            },
        )
    )

    assert list(result.output_artifact_refs) == ["ocr_pages", "ocr_document"]
    page_result_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "ocr.page_result"
    ]
    document_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "ocr.document_result"
    ]
    request_trace_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "ocr.request_trace"
    ]
    response_trace_artifacts = [
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "ocr.response_trace"
    ]
    output_sequence = result.artifact_sequences[0]
    assert output_sequence.artifact_type == "ocr.page_result"
    assert output_sequence.index_key == "page_number"
    assert output_sequence.item_refs == [
        artifact.ref() for artifact in page_result_artifacts
    ]
    assert [artifact.artifact_type for artifact in result.artifact_sequences] == [
        "ocr.page_result",
        "ocr.request_trace",
        "ocr.response_trace",
    ]
    assert [artifact.metadata["page_number"] for artifact in page_result_artifacts] == [
        10,
        11,
    ]
    assert [artifact.input_artifact_ids for artifact in page_result_artifacts] == [
        [first_page.id],
        [second_page.id],
    ]
    document_artifact = document_artifacts[0]
    document_payload = OcrDocumentResultPayload.model_validate_json(
        _load_payload_bytes(storage, document_artifact)
    )
    assert document_artifact.input_artifact_ids == [
        artifact.id for artifact in page_result_artifacts
    ]
    assert document_artifact.metadata["page_count"] == 2
    assert document_artifact.metadata["text_length"] == len(
        "first page text\n\nsecond page text"
    )
    assert document_payload.engine == "local.text"
    assert document_payload.page_count == 2
    assert document_payload.text == "first page text\n\nsecond page text"
    assert document_payload.source_page_sequence_id == str(sequence.id)
    assert document_payload.page_result_artifact_ids == [
        str(artifact.id) for artifact in page_result_artifacts
    ]
    assert [artifact.metadata["page_number"] for artifact in request_trace_artifacts] == [
        10,
        11,
    ]
    assert [
        artifact.metadata["page_number"] for artifact in response_trace_artifacts
    ] == [10, 11]
    assert result.output_artifact_refs == {
        "ocr_pages": output_sequence.ref(),
        "ocr_document": document_artifact.ref(),
    }
    assert result.input_assembly_traces[0].selected_inputs == {
        "pages": [first_page.ref(), second_page.ref()]
    }
    assert result.invocation_traces[0].runtime == {
        "page_count": 2,
        "sequence_index": 1,
        "page_number": 10,
        "engine": "local.text",
        "language_hints": ["pl", "la"],
    }
    assert result.invocation_traces[0].request_ref == (
        request_trace_artifacts[0].payload_ref
    )
    assert result.invocation_traces[0].response_ref == (
        response_trace_artifacts[0].payload_ref
    )

    first_result_location = page_result_artifacts[0].payload_ref.removeprefix(
        "artifact://ocr-page-results/"
    )
    first_result = storage.load("ocr-page-results", first_result_location)
    assert b'"text": "first page text"' in first_result.payload
    assert b'"engine": "local.text"' in first_result.payload

    request_trace = OcrRequestTracePayload.model_validate_json(
        _load_payload_bytes(storage, request_trace_artifacts[0])
    )
    response_trace = OcrResponseTracePayload.model_validate_json(
        _load_payload_bytes(storage, response_trace_artifacts[0])
    )
    assert request_trace.engine_config == {
        "api_key": "<redacted>",
        "timeout_seconds": 12,
    }
    assert request_trace.image_artifact_id == str(first_page.id)
    assert request_trace.image_payload_ref == first_page.payload_ref
    assert response_trace.ocr_result_artifact_id == str(page_result_artifacts[0].id)
    assert response_trace.text_length == len("first page text")


@pytest.mark.asyncio
async def test_ocr_extract_page_handler_emits_single_page_result_with_trace_refs(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    stored = storage.save(
        SaveArtifactPayloadCommand(
            bucket="source-page-images",
            key="pages/page-7.png",
            payload=b"single page text",
        )
    )
    workflow_run_id = uuid4()
    source_sequence_id = uuid4()
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="ocr",
        operator_id=OCR_EXTRACT_PAGE_OPERATOR_ID,
        operator_version=OCR_EXTRACT_PAGE_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "engine": "local.text",
                "language_hints": ["pl"],
                "engine_config": {"timeout_seconds": 12},
            },
            "map_item_index": 3,
            "map_source_sequence_id": str(source_sequence_id),
        },
    )
    page = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
        content_hash=stored.sha256,
        metadata={"page_number": 7},
    )

    result = await OcrExtractPageHandler(storage).execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={"pages": [page]},
        )
    )

    page_result_artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "ocr.page_result"
    )
    request_trace_artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "ocr.request_trace"
    )
    response_trace_artifact = next(
        artifact
        for artifact in result.artifacts
        if artifact.artifact_type == "ocr.response_trace"
    )
    page_payload = OcrPageResultPayload.model_validate_json(
        _load_payload_bytes(storage, page_result_artifact)
    )
    request_trace = OcrRequestTracePayload.model_validate_json(
        _load_payload_bytes(storage, request_trace_artifact)
    )
    response_trace = OcrResponseTracePayload.model_validate_json(
        _load_payload_bytes(storage, response_trace_artifact)
    )

    assert result.output_artifact_refs == {
        "ocr_pages": [page_result_artifact.ref()]
    }
    assert result.artifact_sequences == []
    assert page_payload.text == "single page text"
    assert page_payload.page_number == 7
    assert page_payload.runtime["language_hints"] == ["pl"]
    assert page_payload.runtime["engine_config"] == {"timeout_seconds": 12}
    assert page_payload.runtime["source_page_sequence_id"] == str(source_sequence_id)
    assert page_result_artifact.metadata["sequence_index"] == 3
    assert page_result_artifact.metadata["source_page_sequence_id"] == (
        str(source_sequence_id)
    )
    assert page_result_artifact.metadata["request_trace_artifact_ref"] == {
        "artifact_id": str(request_trace_artifact.id),
        "artifact_type": "ocr.request_trace",
        "schema_version": 1,
        "content_hash": request_trace_artifact.content_hash,
    }
    assert page_result_artifact.metadata["response_trace_artifact_ref"] == {
        "artifact_id": str(response_trace_artifact.id),
        "artifact_type": "ocr.response_trace",
        "schema_version": 1,
        "content_hash": response_trace_artifact.content_hash,
    }
    assert request_trace.sequence_index == 3
    assert request_trace.input_sequence_id == str(source_sequence_id)
    assert response_trace.ocr_result_artifact_id == str(page_result_artifact.id)
    assert result.input_assembly_traces[0].selected_inputs == {
        "pages": [page.ref()]
    }
    assert result.invocation_traces[0].invocation_type == OCR_EXTRACT_PAGE_OPERATOR_ID
    assert result.invocation_traces[0].request_ref == request_trace_artifact.payload_ref
    assert result.invocation_traces[0].response_ref == response_trace_artifact.payload_ref


@pytest.mark.asyncio
async def test_ocr_collect_pages_handler_emits_document_from_page_results(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    source_sequence_id = uuid4()
    page_artifacts: list[Artifact] = []
    for sequence_index, text in enumerate(
        ["first collected page", "second collected page"],
        start=1,
    ):
        stored = storage.save(
            SaveArtifactPayloadCommand(
                bucket="source-page-images",
                key=f"pages/page-{sequence_index}.png",
                payload=text.encode("utf-8"),
            )
        )
        source_page = Artifact(
            artifact_type="source.page_image",
            schema_version=1,
            workflow_run_id=None,
            producer_node_run_id=None,
            payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
            content_hash=stored.sha256,
            metadata={"page_number": sequence_index},
        )
        page_node_run = NodeRun(
            workflow_run_id=workflow_run_id,
            workflow_node_id="ocr",
            operator_id=OCR_EXTRACT_PAGE_OPERATOR_ID,
            operator_version=OCR_EXTRACT_PAGE_OPERATOR_VERSION,
            metadata={
                "workflow_node_config": {"engine": "local.text"},
                "map_item_index": sequence_index,
                "map_source_sequence_id": str(source_sequence_id),
            },
        )
        page_result = await OcrExtractPageHandler(storage).execute(
            NodeExecutionRequest(
                node_run=page_node_run,
                input_artifacts={"pages": [source_page]},
            )
        )
        page_artifacts.append(
            next(
                artifact
                for artifact in page_result.artifacts
                if artifact.artifact_type == "ocr.page_result"
            )
        )

    page_sequence = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[artifact.ref() for artifact in page_artifacts],
        index_key="page_number",
    )
    collect_node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="collect",
        operator_id=OCR_COLLECT_PAGES_OPERATOR_ID,
        operator_version=OCR_COLLECT_PAGES_OPERATOR_VERSION,
    )

    result = await OcrCollectPagesHandler(storage).execute(
        NodeExecutionRequest(
            node_run=collect_node_run,
            input_artifacts={
                "ocr_pages": ArtifactSequenceInput(
                    sequence=page_sequence,
                    artifacts=page_artifacts,
                )
            },
        )
    )

    document_artifact = result.artifacts[0]
    document_payload = OcrDocumentResultPayload.model_validate_json(
        _load_payload_bytes(storage, document_artifact)
    )

    assert result.output_artifact_refs == {
        "ocr_pages": page_sequence.ref(),
        "ocr_document": document_artifact.ref(),
    }
    assert [sequence.artifact_type for sequence in result.artifact_sequences] == [
        "ocr.request_trace",
        "ocr.response_trace",
    ]
    assert [len(sequence.item_refs) for sequence in result.artifact_sequences] == [
        2,
        2,
    ]
    assert document_payload.engine == "local.text"
    assert document_payload.page_count == 2
    assert document_payload.text == "first collected page\n\nsecond collected page"
    assert document_payload.source_page_sequence_id == str(source_sequence_id)
    assert document_payload.page_result_artifact_ids == [
        str(artifact.id) for artifact in page_artifacts
    ]
    assert document_artifact.metadata["ocr_page_sequence_id"] == str(page_sequence.id)
    assert result.input_assembly_traces[0].selected_inputs == {
        "ocr_pages": [artifact.ref() for artifact in page_artifacts]
    }
    assert result.invocation_traces[0].invocation_type == OCR_COLLECT_PAGES_OPERATOR_ID


@pytest.mark.asyncio
async def test_ocr_extract_pages_handler_preserves_retryable_engine_failure(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    stored = storage.save(
        SaveArtifactPayloadCommand(
            bucket="source-page-images",
            key="pages/page-1.png",
            payload=b"image-bytes",
        )
    )
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="ocr",
        operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
        operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
        metadata={"workflow_node_config": {"engine": RetryableOcrEngine.engine_id}},
    )
    page = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
    )
    sequence = ArtifactSequence(
        artifact_type="source.page_image",
        schema_version=1,
        item_refs=[page.ref()],
    )

    with pytest.raises(NodeRunExecutionError, match="provider rate limited") as exc:
        await OcrExtractPagesHandler(
            storage,
            engines={RetryableOcrEngine.engine_id: RetryableOcrEngine()},
        ).execute(
            NodeExecutionRequest(
                node_run=node_run,
                input_artifacts={
                    "pages": ArtifactSequenceInput(
                        sequence=sequence,
                        artifacts=[page],
                    )
                },
            )
        )

    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_ocr_compare_pages_handler_emits_page_sequence_and_metrics(
    tmp_path,
) -> None:
    storage = LocalArtifactPayloadStorage(tmp_path)
    workflow_run_id = uuid4()
    first_a = _save_ocr_page_result(
        storage,
        workflow_run_id=workflow_run_id,
        key="a/page-1.json",
        page_number=1,
        engine="local.text",
        text="Alpha page",
    )
    second_a = _save_ocr_page_result(
        storage,
        workflow_run_id=workflow_run_id,
        key="a/page-2.json",
        page_number=2,
        engine="local.text",
        text="Beta page",
    )
    first_b = _save_ocr_page_result(
        storage,
        workflow_run_id=workflow_run_id,
        key="b/page-1.json",
        page_number=1,
        engine="mistral.ocr",
        text="Alpha page",
    )
    second_b = _save_ocr_page_result(
        storage,
        workflow_run_id=workflow_run_id,
        key="b/page-2.json",
        page_number=2,
        engine="mistral.ocr",
        text="Beta page corrected",
    )
    sequence_a = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[first_a.ref(), second_a.ref()],
        index_key="page_number",
    )
    sequence_b = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[first_b.ref(), second_b.ref()],
        index_key="page_number",
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="compare",
        operator_id=OCR_COMPARE_PAGES_OPERATOR_ID,
        operator_version=OCR_COMPARE_PAGES_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "candidate_a_label": "Local",
                "candidate_b_label": "Mistral",
            }
        },
    )

    result = await OcrComparePagesHandler(storage).execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={
                "candidate_a_pages": ArtifactSequenceInput(
                    sequence=sequence_a,
                    artifacts=[first_a, second_a],
                ),
                "candidate_b_pages": ArtifactSequenceInput(
                    sequence=sequence_b,
                    artifacts=[first_b, second_b],
                ),
            },
        )
    )

    comparison_sequence = result.artifact_sequences[0]
    metrics_artifact = result.artifacts[-1]
    comparison_artifacts = result.artifacts[:-1]
    assert list(result.output_artifact_refs) == ["comparison_pages", "metrics"]
    assert result.output_artifact_refs == {
        "comparison_pages": comparison_sequence.ref(),
        "metrics": metrics_artifact.ref(),
    }
    assert comparison_sequence.artifact_type == "ocr.comparison_result"
    assert comparison_sequence.item_refs == [
        artifact.ref() for artifact in comparison_artifacts
    ]
    assert [artifact.artifact_type for artifact in comparison_artifacts] == [
        "ocr.comparison_result",
        "ocr.comparison_result",
    ]
    assert metrics_artifact.artifact_type == "evaluation.metrics"
    assert comparison_artifacts[0].input_artifact_ids == [first_a.id, first_b.id]
    assert comparison_artifacts[1].metadata["candidate_a_engine"] == "local.text"
    assert comparison_artifacts[1].metadata["candidate_b_engine"] == "mistral.ocr"

    first_comparison = _load_comparison_page(storage, comparison_artifacts[0])
    second_comparison = _load_comparison_page(storage, comparison_artifacts[1])
    metrics = _load_comparison_metrics(storage, metrics_artifact)
    assert first_comparison.candidate_a_label == "Local"
    assert first_comparison.candidate_b_label == "Mistral"
    assert first_comparison.sequence_index == 1
    assert first_comparison.equal_text is True
    assert first_comparison.similarity_ratio == 1.0
    assert second_comparison.sequence_index == 2
    assert second_comparison.equal_text is False
    assert second_comparison.similarity_ratio < 1.0
    assert metrics.metric_family == "ocr_comparison"
    assert metrics.metrics["page_count"] == 2
    assert metrics.metrics["mean_similarity_ratio"] == pytest.approx(
        (first_comparison.similarity_ratio + second_comparison.similarity_ratio) / 2
    )
    assert metrics.source_artifact_ids == [
        str(first_a.id),
        str(second_a.id),
        str(first_b.id),
        str(second_b.id),
    ]
    assert result.input_assembly_traces[0].policies == {
        "similarity_algorithm": "difflib.SequenceMatcher",
        "autojunk": False,
    }
    assert result.invocation_traces[0].runtime == {
        "page_count": 2,
        "mean_similarity_ratio": metrics.metrics["mean_similarity_ratio"],
    }


@pytest.mark.asyncio
async def test_ocr_compare_pages_handler_rejects_mismatched_sequence_counts(
    tmp_path,
) -> None:
    first = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=uuid4(),
        producer_node_run_id=None,
        payload_ref="artifact://ocr-page-results/a/page-1.json",
    )
    second = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=uuid4(),
        producer_node_run_id=None,
        payload_ref="artifact://ocr-page-results/a/page-2.json",
    )
    other = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=uuid4(),
        producer_node_run_id=None,
        payload_ref="artifact://ocr-page-results/b/page-1.json",
    )
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="compare",
        operator_id=OCR_COMPARE_PAGES_OPERATOR_ID,
        operator_version=OCR_COMPARE_PAGES_OPERATOR_VERSION,
    )

    with pytest.raises(NodeRunExecutionError, match="matching page counts"):
        await OcrComparePagesHandler(LocalArtifactPayloadStorage(tmp_path)).execute(
            NodeExecutionRequest(
                node_run=node_run,
                input_artifacts={
                    "candidate_a_pages": ArtifactSequenceInput(
                        sequence=ArtifactSequence(
                            artifact_type="ocr.page_result",
                            schema_version=1,
                            item_refs=[first.ref(), second.ref()],
                        ),
                        artifacts=[first, second],
                    ),
                    "candidate_b_pages": ArtifactSequenceInput(
                        sequence=ArtifactSequence(
                            artifact_type="ocr.page_result",
                            schema_version=1,
                            item_refs=[other.ref()],
                        ),
                        artifacts=[other],
                    ),
                },
            )
        )


@pytest.mark.asyncio
async def test_ocr_select_pages_handler_emits_selected_sequence() -> None:
    workflow_run_id = uuid4()
    first_a = _ocr_artifact(
        workflow_run_id=workflow_run_id,
        page_number=1,
        engine="local.text",
    )
    second_a = _ocr_artifact(
        workflow_run_id=workflow_run_id,
        page_number=2,
        engine="local.text",
    )
    first_b = _ocr_artifact(
        workflow_run_id=workflow_run_id,
        page_number=1,
        engine="mistral.ocr",
    )
    second_b = _ocr_artifact(
        workflow_run_id=workflow_run_id,
        page_number=2,
        engine="mistral.ocr",
    )
    comparison = Artifact(
        artifact_type="ocr.comparison_result",
        schema_version=1,
        workflow_run_id=workflow_run_id,
        producer_node_run_id=None,
        payload_ref="artifact://ocr-comparison-results/comparison/page-1.json",
    )
    sequence_a = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[first_a.ref(), second_a.ref()],
        index_key="page_number",
    )
    sequence_b = ArtifactSequence(
        artifact_type="ocr.page_result",
        schema_version=1,
        item_refs=[first_b.ref(), second_b.ref()],
        index_key="page_number",
    )
    comparison_sequence = ArtifactSequence(
        artifact_type="ocr.comparison_result",
        schema_version=1,
        item_refs=[comparison.ref(), comparison.ref()],
        index_key="page_number",
    )
    node_run = NodeRun(
        workflow_run_id=workflow_run_id,
        workflow_node_id="select",
        operator_id=OCR_SELECT_PAGES_OPERATOR_ID,
        operator_version=OCR_SELECT_PAGES_OPERATOR_VERSION,
        metadata={
            "workflow_node_config": {
                "selected_candidate": "candidate_b",
                "decision_note": "Mistral preserved table headers",
            }
        },
    )

    result = await OcrSelectPagesHandler().execute(
        NodeExecutionRequest(
            node_run=node_run,
            input_artifacts={
                "candidate_a_pages": ArtifactSequenceInput(
                    sequence=sequence_a,
                    artifacts=[first_a, second_a],
                ),
                "candidate_b_pages": ArtifactSequenceInput(
                    sequence=sequence_b,
                    artifacts=[first_b, second_b],
                ),
                "comparison_pages": ArtifactSequenceInput(
                    sequence=comparison_sequence,
                    artifacts=[comparison, comparison],
                ),
            },
        )
    )

    selected_sequence = result.artifact_sequences[0]
    assert result.artifacts == []
    assert result.output_artifact_refs == {"selected_pages": selected_sequence.ref()}
    assert selected_sequence.artifact_type == "ocr.page_result"
    assert selected_sequence.item_refs == [first_b.ref(), second_b.ref()]
    assert selected_sequence.index_key == "page_number"
    assert selected_sequence.metadata == {
        "selected_candidate": "candidate_b",
        "selected_sequence_id": str(sequence_b.id),
        "rejected_sequence_id": str(sequence_a.id),
        "page_count": 2,
        "comparison_sequence_id": str(comparison_sequence.id),
        "decision_note": "Mistral preserved table headers",
    }
    assert result.input_assembly_traces[0].policies == {
        "selected_candidate": "candidate_b"
    }
    assert result.invocation_traces[0].runtime == {
        "selected_candidate": "candidate_b",
        "page_count": 2,
    }
    assert result.invocation_traces[0].output_artifact_refs == [
        first_b.ref(),
        second_b.ref(),
    ]


@pytest.mark.asyncio
async def test_ocr_select_pages_handler_rejects_invalid_selected_candidate() -> None:
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="select",
        operator_id=OCR_SELECT_PAGES_OPERATOR_ID,
        operator_version=OCR_SELECT_PAGES_OPERATOR_VERSION,
        metadata={"workflow_node_config": {"selected_candidate": "candidate_c"}},
    )

    with pytest.raises(NodeRunExecutionError, match="candidate_a or candidate_b"):
        await OcrSelectPagesHandler().execute(
            NodeExecutionRequest(node_run=node_run, input_artifacts={})
        )


@pytest.mark.asyncio
async def test_ocr_select_pages_handler_rejects_mismatched_sequence_counts() -> None:
    first_a = _ocr_artifact(
        workflow_run_id=uuid4(),
        page_number=1,
        engine="local.text",
    )
    second_a = _ocr_artifact(
        workflow_run_id=uuid4(),
        page_number=2,
        engine="local.text",
    )
    first_b = _ocr_artifact(
        workflow_run_id=uuid4(),
        page_number=1,
        engine="mistral.ocr",
    )
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="select",
        operator_id=OCR_SELECT_PAGES_OPERATOR_ID,
        operator_version=OCR_SELECT_PAGES_OPERATOR_VERSION,
    )

    with pytest.raises(NodeRunExecutionError, match="matching page counts"):
        await OcrSelectPagesHandler().execute(
            NodeExecutionRequest(
                node_run=node_run,
                input_artifacts={
                    "candidate_a_pages": ArtifactSequenceInput(
                        sequence=ArtifactSequence(
                            artifact_type="ocr.page_result",
                            schema_version=1,
                            item_refs=[first_a.ref(), second_a.ref()],
                        ),
                        artifacts=[first_a, second_a],
                    ),
                    "candidate_b_pages": ArtifactSequenceInput(
                        sequence=ArtifactSequence(
                            artifact_type="ocr.page_result",
                            schema_version=1,
                            item_refs=[first_b.ref()],
                        ),
                        artifacts=[first_b],
                    ),
                },
            )
        )


def test_tesseract_ocr_engine_maps_word_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_image_to_data(
        image: Image.Image,
        output_type: object,
        lang: str,
        config: str,
    ) -> dict[str, list[object]]:
        captured["mode"] = image.mode
        captured["size"] = image.size
        captured["output_type"] = output_type
        captured["lang"] = lang
        captured["config"] = config
        return {
            "text": ["", "Alpha", "Beta", "ignored"],
            "conf": ["-1", "96.0", "84", "-1"],
            "left": [0, 10, 50, 0],
            "top": [0, 5, 20, 0],
            "width": [0, 20, 30, 0],
            "height": [0, 10, 10, 0],
        }

    monkeypatch.setattr(
        worker_operators.pytesseract,
        "image_to_data",
        fake_image_to_data,
    )
    source_image = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/pages/page-1.png",
    )

    result = TesseractOcrEngine().extract_page(
        OcrPageInput(
            page_number=7,
            image_artifact=source_image,
            payload=_png_bytes(width=100, height=50),
            language_hints=("lat", "pol"),
            engine_config={"psm": 11, "oem": 1},
        )
    )

    assert captured["mode"] == "RGB"
    assert captured["size"] == (100, 50)
    assert captured["lang"] == "lat+pol"
    assert captured["config"] == "--psm 11 --oem 1"
    assert result.page_number == 7
    assert result.engine == "local.tesseract"
    assert result.text == "Alpha Beta"
    assert result.confidence == 90.0
    assert result.tokens == [
        {
            "text": "Alpha",
            "confidence": 96.0,
            "bbox": [10, 5, 30, 15],
            "normalized_bbox": [100, 100, 300, 300],
        },
        {
            "text": "Beta",
            "confidence": 84.0,
            "bbox": [50, 20, 80, 30],
            "normalized_bbox": [500, 400, 800, 600],
        },
    ]
    assert result.runtime["language"] == "lat+pol"
    assert result.runtime["psm"] == 11
    assert result.runtime["oem"] == 1


def test_tesseract_ocr_engine_prefers_configured_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_image_to_data(
        image: Image.Image,
        output_type: object,
        lang: str,
        config: str,
    ) -> dict[str, list[object]]:
        captured["lang"] = lang
        return {"text": ["Alpha"], "conf": ["99"], "left": [0], "top": [0], "width": [1], "height": [1]}

    monkeypatch.setattr(
        worker_operators.pytesseract,
        "image_to_data",
        fake_image_to_data,
    )
    source_image = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/pages/page-1.png",
    )

    TesseractOcrEngine().extract_page(
        OcrPageInput(
            page_number=1,
            image_artifact=source_image,
            payload=_png_bytes(),
            language_hints=("lat", "pol"),
            engine_config={"language": "eng"},
        )
    )

    assert captured["lang"] == "eng"


def test_openai_compatible_structured_extraction_engine_posts_schema_and_maps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        payload = json.loads(request.content.decode("utf-8"))
        captured["payload"] = payload
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"text": "Alpha page", "page_number": 3}
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setenv("OPENAI_COMPAT_TEST_API_KEY", "secret-token")
    engine = OpenAICompatibleStructuredExtractionEngine(
        httpx.Client(transport=httpx.MockTransport(handler))
    )
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "page_number": {"type": "integer"},
        },
        "required": ["text", "page_number"],
        "additionalProperties": False,
    }

    result = engine.extract_page(
        ContextualExtractionPageInput(
            sequence_index=1,
            page_number=3,
            page_text="Alpha page",
            rendered_prompt="Extract Alpha page",
            context={},
            schema=schema,
            result_key="record",
        ),
        ModelBindingPayload(
            provider="openai-compatible",
            model="test-model",
            parameters={
                "api_key_env_var": "OPENAI_COMPAT_TEST_API_KEY",
                "base_url": "https://llm.example/v1",
                "schema_name": "page_record",
                "temperature": 0,
                "max_tokens": 512,
            },
            capabilities=["structured_output"],
        ),
    )

    request_payload = captured["payload"]
    assert captured["url"] == "https://llm.example/v1/chat/completions"
    assert captured["authorization"] == "Bearer secret-token"
    assert request_payload["model"] == "test-model"
    assert request_payload["messages"][1]["content"] == "Extract Alpha page"
    assert request_payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "page_record",
            "schema": schema,
            "strict": True,
        },
    }
    assert request_payload["max_tokens"] == 512
    assert "secret-token" not in json.dumps(request_payload)
    assert result == {"text": "Alpha page", "page_number": 3}


def test_openai_compatible_structured_extraction_engine_marks_rate_limit_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            request=request,
            json={"error": "limit"},
        )

    monkeypatch.setenv("OPENAI_COMPAT_TEST_API_KEY", "secret-token")
    engine = OpenAICompatibleStructuredExtractionEngine(
        httpx.Client(transport=httpx.MockTransport(handler))
    )

    with pytest.raises(StructuredExtractionEngineError, match="status 429") as exc:
        engine.extract_page(
            ContextualExtractionPageInput(
                sequence_index=1,
                page_number=2,
                page_text="Alpha page",
                rendered_prompt="Extract Alpha page",
                context={},
                schema={"type": "object"},
                result_key="record",
            ),
            ModelBindingPayload(
                provider="openai-compatible",
                model="test-model",
                parameters={
                    "api_key_env_var": "OPENAI_COMPAT_TEST_API_KEY",
                    "base_url": "https://llm.example/v1",
                },
            ),
        )

    assert exc.value.retryable is True
    assert "page 2" in str(exc.value)
    assert "secret-token" not in str(exc.value)


def test_mistral_ocr_engine_posts_image_data_url_and_maps_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers["authorization"]
        payload = json.loads(request.content.decode("utf-8"))
        captured["payload"] = payload
        return httpx.Response(
            status_code=200,
            json={
                "pages": [
                    {
                        "index": 0,
                        "markdown": "Alpha **Beta**",
                        "blocks": [
                            {"type": "text", "text": "Alpha"},
                            "not-a-block",
                        ],
                        "confidence": 0.87,
                    }
                ]
            },
        )

    monkeypatch.setenv("MISTRAL_TEST_API_KEY", "secret-token")
    source_image = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/pages/page-1.jpg",
        metadata={"content_type": "image/jpeg"},
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))

    result = MistralOcrEngine(client).extract_page(
        OcrPageInput(
            page_number=3,
            image_artifact=source_image,
            payload=b"jpeg-bytes",
            engine_config={
                "api_key_env_var": "MISTRAL_TEST_API_KEY",
                "base_url": "https://mistral.example",
                "model": "mistral-ocr-latest",
                "include_blocks": ["text", "image"],
                "timeout_seconds": 12,
            },
        )
    )

    request_payload = captured["payload"]
    assert captured["url"] == "https://mistral.example/v1/ocr"
    assert captured["authorization"] == "Bearer secret-token"
    assert request_payload["model"] == "mistral-ocr-latest"
    assert request_payload["include_blocks"] == ["text", "image"]
    assert request_payload["document"]["type"] == "image_url"
    assert request_payload["document"]["image_url"].startswith(
        "data:image/jpeg;base64,"
    )
    assert result.engine == "mistral.ocr"
    assert result.text == "Alpha **Beta**"
    assert result.blocks == [{"type": "text", "text": "Alpha"}]
    assert result.confidence == 0.87
    assert result.runtime == {
        "byte_size": 10,
        "model": "mistral-ocr-latest",
        "base_url": "https://mistral.example",
        "include_blocks": ["text", "image"],
        "provider_page_count": 1,
    }


def test_mistral_ocr_engine_requires_api_key_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    source_image = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/pages/page-1.png",
    )

    with pytest.raises(ValueError, match="MISTRAL_API_KEY is required"):
        MistralOcrEngine().extract_page(
            OcrPageInput(
                page_number=1,
                image_artifact=source_image,
                payload=b"image-bytes",
            )
        )


def test_mistral_ocr_engine_marks_rate_limit_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=429, request=request, json={"error": "limit"})

    monkeypatch.setenv("MISTRAL_TEST_API_KEY", "secret-token")
    source_image = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/pages/page-1.png",
    )

    with pytest.raises(OcrPageEngineError, match="status 429") as exc:
        MistralOcrEngine(
            httpx.Client(transport=httpx.MockTransport(handler))
        ).extract_page(
            OcrPageInput(
                page_number=1,
                image_artifact=source_image,
                payload=b"image-bytes",
                engine_config={
                    "api_key_env_var": "MISTRAL_TEST_API_KEY",
                    "base_url": "https://mistral.example",
                },
            )
        )

    assert exc.value.retryable is True
    assert "mistral-ocr-latest" in str(exc.value)
    assert "https://mistral.example" in str(exc.value)
    assert "secret-token" not in str(exc.value)


def test_mistral_ocr_engine_marks_auth_failure_as_permanent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=401, request=request, json={"error": "auth"})

    monkeypatch.setenv("MISTRAL_TEST_API_KEY", "secret-token")
    source_image = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/pages/page-1.png",
    )

    with pytest.raises(OcrPageEngineError, match="status 401") as exc:
        MistralOcrEngine(
            httpx.Client(transport=httpx.MockTransport(handler))
        ).extract_page(
            OcrPageInput(
                page_number=1,
                image_artifact=source_image,
                payload=b"image-bytes",
                engine_config={
                    "api_key_env_var": "MISTRAL_TEST_API_KEY",
                    "base_url": "https://mistral.example",
                },
            )
        )

    assert exc.value.retryable is False


def test_mistral_ocr_engine_marks_transport_failure_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down", request=request)

    monkeypatch.setenv("MISTRAL_TEST_API_KEY", "secret-token")
    source_image = Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=None,
        producer_node_run_id=None,
        payload_ref="artifact://source-page-images/pages/page-1.png",
    )

    with pytest.raises(OcrPageEngineError, match="transport failed") as exc:
        MistralOcrEngine(
            httpx.Client(transport=httpx.MockTransport(handler))
        ).extract_page(
            OcrPageInput(
                page_number=1,
                image_artifact=source_image,
                payload=b"image-bytes",
                engine_config={
                    "api_key_env_var": "MISTRAL_TEST_API_KEY",
                    "base_url": "https://mistral.example",
                },
            )
        )

    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_ocr_extract_pages_handler_rejects_unregistered_engine(tmp_path) -> None:
    node_run = NodeRun(
        workflow_run_id=uuid4(),
        workflow_node_id="ocr",
        operator_id=OCR_EXTRACT_PAGES_OPERATOR_ID,
        operator_version=OCR_EXTRACT_PAGES_OPERATOR_VERSION,
        metadata={"workflow_node_config": {"engine": "missing"}},
    )

    with pytest.raises(NodeRunExecutionError, match="no OCR engine registered"):
        await OcrExtractPagesHandler(LocalArtifactPayloadStorage(tmp_path)).execute(
            NodeExecutionRequest(node_run=node_run, input_artifacts={})
        )


def test_create_app_wires_builtin_node_handlers_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured_handlers: list[Mapping[tuple[str, str], NodeRunHandler]] = []
    captured_node_specs: list[Mapping[tuple[str, str], object]] = []

    class CapturingNodeRunExecutor:
        def __init__(
            self,
            uow_factory: Callable[[], StudioUnitOfWorkPort],
            handlers: Mapping[tuple[str, str], NodeRunHandler],
            node_specs: Mapping[tuple[str, str], object],
        ) -> None:
            captured_handlers.append(handlers)
            captured_node_specs.append(node_specs)

        async def execute_node_run(self, node_run_id: UUID | str) -> None:
            raise AssertionError("subscriber callback should not run during app creation")

    monkeypatch.setattr(streaming, "NodeRunExecutor", CapturingNodeRunExecutor)

    streaming.create_app(
        nats_url="nats://localhost:4222",
        outbox_interval_seconds=999.0,
        payload_storage=LocalArtifactPayloadStorage(tmp_path),
    )

    assert len(captured_handlers) == 1
    assert len(captured_node_specs) == 1
    assert (
        DEBUG_EMIT_TEXT_OPERATOR_ID,
        DEBUG_EMIT_TEXT_OPERATOR_VERSION,
    ) in captured_node_specs[0]
    assert (
        DEBUG_EMIT_TEXT_OPERATOR_ID,
        DEBUG_EMIT_TEXT_OPERATOR_VERSION,
    ) in captured_handlers[0]
    assert (
        OCR_COMPARE_PAGES_OPERATOR_ID,
        OCR_COMPARE_PAGES_OPERATOR_VERSION,
    ) in captured_handlers[0]
    assert (
        OCR_EXTRACT_PAGES_OPERATOR_ID,
        OCR_EXTRACT_PAGES_OPERATOR_VERSION,
    ) in captured_handlers[0]


def _save_ocr_page_result(
    storage: LocalArtifactPayloadStorage,
    *,
    workflow_run_id: UUID,
    key: str,
    page_number: int,
    engine: str,
    text: str,
) -> Artifact:
    payload = OcrPageResultPayload(
        page_number=page_number,
        engine=engine,
        text=text,
        image_artifact_id=str(uuid4()),
    )
    stored = storage.save(
        SaveArtifactPayloadCommand(
            bucket="ocr-page-results",
            key=key,
            payload=payload.model_dump_json(indent=2).encode("utf-8"),
        )
    )
    return Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=workflow_run_id,
        producer_node_run_id=None,
        payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
        content_hash=stored.sha256,
        metadata={
            "page_number": page_number,
            "engine": engine,
            "text_length": len(text),
        },
    )


def _ocr_artifact(
    *,
    workflow_run_id: UUID,
    page_number: int,
    engine: str,
) -> Artifact:
    return Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=workflow_run_id,
        producer_node_run_id=None,
        payload_ref=f"artifact://ocr-page-results/{engine}/page-{page_number}.json",
        metadata={
            "page_number": page_number,
            "engine": engine,
        },
    )


def _page_image_artifact(
    *,
    workflow_run_id: UUID,
    page_number: int,
) -> Artifact:
    return Artifact(
        artifact_type="source.page_image",
        schema_version=1,
        workflow_run_id=workflow_run_id,
        producer_node_run_id=None,
        payload_ref=f"artifact://source-page-images/page-{page_number}.png",
        metadata={"page_number": page_number},
    )


def _save_payload_artifact(
    storage: LocalArtifactPayloadStorage,
    *,
    workflow_run_id: UUID,
    artifact_type: str,
    bucket: str,
    key: str,
    payload: BaseModel,
) -> Artifact:
    stored = storage.save(
        SaveArtifactPayloadCommand(
            bucket=bucket,
            key=key,
            payload=payload.model_dump_json(indent=2).encode("utf-8"),
        )
    )
    return Artifact(
        artifact_type=artifact_type,
        schema_version=1,
        workflow_run_id=workflow_run_id,
        producer_node_run_id=None,
        payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
        content_hash=stored.sha256,
        metadata={"content_type": "application/json"},
    )


def _load_comparison_page(
    storage: LocalArtifactPayloadStorage,
    artifact: Artifact,
) -> OcrComparisonPagePayload:
    location = parse_artifact_payload_ref(artifact.payload_ref)
    stored = storage.load(location.bucket, location.key)
    return OcrComparisonPagePayload.model_validate_json(stored.payload)


def _load_comparison_metrics(
    storage: LocalArtifactPayloadStorage,
    artifact: Artifact,
) -> EvaluationMetricsPayload:
    location = parse_artifact_payload_ref(artifact.payload_ref)
    stored = storage.load(location.bucket, location.key)
    return EvaluationMetricsPayload.model_validate_json(stored.payload)


def _load_payload_bytes(
    storage: LocalArtifactPayloadStorage,
    artifact: Artifact,
) -> bytes:
    location = parse_artifact_payload_ref(artifact.payload_ref)
    return storage.load(location.bucket, location.key).payload


def _png_bytes(width: int = 10, height: int = 10) -> bytes:
    image = Image.new("RGB", (width, height), color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
