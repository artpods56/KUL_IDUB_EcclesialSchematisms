from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import base64
import csv
from io import BytesIO
from io import StringIO
import json
import os
from typing import Protocol, TypeVar
from uuid import UUID

import httpx
from jinja2 import Environment, StrictUndefined, TemplateError
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaError
import pytesseract
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

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
from notarius_core.domain.models import (
    Artifact,
    ArtifactRef,
    ArtifactSequence,
    InputAssemblyTrace,
    InvocationTrace,
)
from notarius_storage import (
    ArtifactPayloadStoragePort,
    SaveArtifactPayloadCommand,
    artifact_payload_ref,
    parse_artifact_payload_ref,
)
from notarius_worker.node_execution import (
    ArtifactSequenceInput,
    NodeExecutionRequest,
    NodeExecutionResult,
    NodeRunExecutionError,
    NodeRunHandler,
)

PayloadModelT = TypeVar("PayloadModelT", bound=BaseModel)


class EmitTextHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = request.node_run.metadata.get("workflow_node_config", {})
        if not isinstance(raw_config, dict):
            raise NodeRunExecutionError(
                "debug.emit_text expected workflow_node_config metadata to be an object",
                retryable=False,
            )

        raw_text = raw_config.get("text")
        if not isinstance(raw_text, str):
            raise NodeRunExecutionError(
                "debug.emit_text requires workflow_node_config.text to be a string",
                retryable=False,
            )
        if raw_text == "":
            raise NodeRunExecutionError(
                "debug.emit_text requires workflow_node_config.text to be non-empty",
                retryable=False,
            )

        raw_payload_ref = raw_config.get("payload_ref")
        if raw_payload_ref is not None and not isinstance(raw_payload_ref, str):
            raise NodeRunExecutionError(
                "debug.emit_text requires workflow_node_config.payload_ref to be a string",
                retryable=False,
            )
        if raw_payload_ref == "":
            raise NodeRunExecutionError(
                "debug.emit_text requires workflow_node_config.payload_ref to be non-empty",
                retryable=False,
            )
        payload_ref = (
            raw_payload_ref
            if raw_payload_ref is not None
            else f"memory://debug.emit_text/{request.node_run.id}.txt"
        )

        artifact = Artifact(
            artifact_type="debug.text",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            payload_ref=payload_ref,
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            metadata={"text": raw_text},
        )
        invocation_trace = InvocationTrace(
            node_run_id=request.node_run.id,
            invocation_type=DEBUG_EMIT_TEXT_OPERATOR_ID,
            output_artifact_refs=[artifact.ref()],
            runtime={"text_length": len(raw_text)},
        )
        return NodeExecutionResult(
            output_artifact_refs={"text": artifact.ref()},
            artifacts=[artifact],
            invocation_traces=[invocation_trace],
        )


class OcrPageResultPayload(BaseModel):
    page_number: int
    engine: str
    text: str
    blocks: list[dict[str, object]] = Field(default_factory=list)
    tokens: list[dict[str, object]] = Field(default_factory=list)
    confidence: float | None = None
    image_artifact_id: str
    runtime: dict[str, object] = Field(default_factory=dict)


class OcrDocumentResultPayload(BaseModel):
    engine: str
    page_count: int
    text: str
    page_result_artifact_ids: list[str]
    request_trace_sequence_id: str
    response_trace_sequence_id: str
    source_page_sequence_id: str
    language_hints: list[str] = Field(default_factory=list)
    runtime: dict[str, object] = Field(default_factory=dict)


class OcrRequestTracePayload(BaseModel):
    sequence_index: int
    page_number: int
    engine: str
    image_artifact_id: str
    input_sequence_id: str
    image_payload_ref: str
    image_media_type: str
    payload_byte_size: int
    language_hints: list[str] = Field(default_factory=list)
    engine_config: dict[str, object] = Field(default_factory=dict)


class OcrResponseTracePayload(BaseModel):
    sequence_index: int
    page_number: int
    engine: str
    image_artifact_id: str
    ocr_result_artifact_id: str
    text_length: int
    block_count: int
    token_count: int
    confidence: float | None = None
    runtime: dict[str, object] = Field(default_factory=dict)


class OcrComparisonPagePayload(BaseModel):
    sequence_index: int
    candidate_a_page_number: int
    candidate_b_page_number: int
    candidate_a_label: str
    candidate_b_label: str
    candidate_a_artifact_id: str
    candidate_b_artifact_id: str
    candidate_a_engine: str
    candidate_b_engine: str
    candidate_a_image_artifact_id: str
    candidate_b_image_artifact_id: str
    candidate_a_text_length: int
    candidate_b_text_length: int
    similarity_ratio: float
    equal_text: bool


class EvaluationMetricsPayload(BaseModel):
    metric_family: str
    metrics: dict[str, object] = Field(default_factory=dict)
    source_artifact_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class PromptTemplatePayload(BaseModel):
    name: str
    template: str
    template_format: str = "jinja2"
    variables: list[str] = Field(default_factory=list)
    description: str | None = None


class ExtractionSchemaPayload(BaseModel):
    name: str
    json_schema: dict[str, object]
    schema_format: str = "json_schema"
    description: str | None = None


class ModelBindingPayload(BaseModel):
    provider: str
    model: str
    parameters: dict[str, object] = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    credential_ref: str | None = None
    endpoint_ref: str | None = None


class InputPolicyPayload(BaseModel):
    name: str
    policy_type: str
    settings: dict[str, object] = Field(default_factory=dict)
    applies_to: list[str] = Field(default_factory=list)
    description: str | None = None


class ContextBundlePayload(BaseModel):
    name: str
    context: dict[str, object]
    applies_to: list[str] = Field(default_factory=list)
    description: str | None = None


class ModelInputPayload(BaseModel):
    sequence_index: int
    page_number: int
    rendered_prompt: str
    context: dict[str, object]
    prompt_template_artifact_id: str
    extraction_schema_artifact_id: str
    model_binding_artifact_id: str
    input_policy_artifact_id: str
    ocr_artifact_id: str
    context_bundle_artifact_id: str | None = None
    page_artifact_id: str | None = None


class ModelResponsePayload(BaseModel):
    sequence_index: int
    page_number: int
    provider: str
    model: str
    engine: str
    response: dict[str, object]
    validation_errors: list[str] = Field(default_factory=list)
    model_input_artifact_id: str


class ExtractionRecordResultPayload(BaseModel):
    sequence_index: int
    page_number: int
    record: dict[str, object]
    validation_errors: list[str] = Field(default_factory=list)
    evidence: list[dict[str, object]] = Field(default_factory=list)
    model_input_artifact_id: str
    model_response_artifact_id: str
    ocr_artifact_id: str
    page_artifact_id: str | None = None


class ExtractionDocumentResultPayload(BaseModel):
    page_count: int
    record_count: int
    records: list[dict[str, object]]
    validation_error_count: int
    page_result_artifact_ids: list[str]
    model_input_sequence_id: str
    model_response_sequence_id: str
    provider: str
    model: str
    policy_type: str


class ValidationResultPayload(BaseModel):
    source_artifact_id: str
    schema_artifact_id: str
    record_count: int
    valid_record_count: int
    invalid_record_count: int
    valid: bool
    error_count: int
    errors: list[dict[str, object]] = Field(default_factory=list)


class ExportDatasetPayload(BaseModel):
    format: str
    source_artifact_id: str
    record_count: int
    records: list[dict[str, object]]
    metadata: dict[str, object] = Field(default_factory=dict)


def builtin_artifact_payload_models() -> dict[tuple[str, int], type[BaseModel]]:
    return {
        ("context.bundle", 1): ContextBundlePayload,
        ("evaluation.metrics", 1): EvaluationMetricsPayload,
        ("export.dataset", 1): ExportDatasetPayload,
        ("extraction.document_result", 1): ExtractionDocumentResultPayload,
        ("extraction.record_result", 1): ExtractionRecordResultPayload,
        ("extraction.schema", 1): ExtractionSchemaPayload,
        ("input.policy", 1): InputPolicyPayload,
        ("model.binding", 1): ModelBindingPayload,
        ("model.input", 1): ModelInputPayload,
        ("model.response", 1): ModelResponsePayload,
        ("ocr.comparison_result", 1): OcrComparisonPagePayload,
        ("ocr.document_result", 1): OcrDocumentResultPayload,
        ("ocr.page_result", 1): OcrPageResultPayload,
        ("ocr.request_trace", 1): OcrRequestTracePayload,
        ("ocr.response_trace", 1): OcrResponseTracePayload,
        ("prompt.template", 1): PromptTemplatePayload,
        ("validation.result", 1): ValidationResultPayload,
    }


@dataclass(frozen=True, slots=True)
class OcrPageInput:
    page_number: int
    image_artifact: Artifact
    payload: bytes
    language_hints: tuple[str, ...] = ()
    engine_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OcrPageEngineError(RuntimeError):
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


class OcrPageEnginePort(Protocol):
    engine_id: str

    def extract_page(self, page: OcrPageInput) -> OcrPageResultPayload: ...


@dataclass(frozen=True, slots=True)
class ContextualExtractionPageInput:
    sequence_index: int
    page_number: int
    page_text: str
    rendered_prompt: str
    context: dict[str, object]
    schema: dict[str, object]
    result_key: str


class StructuredExtractionEnginePort(Protocol):
    engine_id: str

    def extract_page(
        self,
        page: ContextualExtractionPageInput,
        binding: ModelBindingPayload,
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class StructuredExtractionEngineError(RuntimeError):
    message: str
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


class LocalEchoStructuredExtractionEngine:
    engine_id = "local.echo"

    def extract_page(
        self,
        page: ContextualExtractionPageInput,
        binding: ModelBindingPayload,
    ) -> dict[str, object]:
        properties = page.schema.get("properties", {})
        if isinstance(properties, dict):
            result: dict[str, object] = {}
            for key in properties:
                if key == "text":
                    result[key] = page.page_text
                elif key == "page_number":
                    result[key] = page.page_number
                elif key == "sequence_index":
                    result[key] = page.sequence_index
                elif key == "summary":
                    result[key] = page.page_text[:240]
                elif key == "context":
                    result[key] = page.context
        else:
            result = {}

        if not result:
            result = {
                page.result_key: {
                    "text": page.page_text,
                    "page_number": page.page_number,
                    "sequence_index": page.sequence_index,
                }
            }
        return result


class OpenAICompatibleStructuredExtractionEngine:
    engine_id = "openai-compatible"

    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self.http_client = http_client

    def extract_page(
        self,
        page: ContextualExtractionPageInput,
        binding: ModelBindingPayload,
    ) -> dict[str, object]:
        api_key_env_var = _string_model_parameter(
            binding.parameters,
            key="api_key_env_var",
            default="OPENAI_API_KEY",
        )
        api_key = os.getenv(api_key_env_var)
        if api_key is None or api_key == "":
            raise StructuredExtractionEngineError(
                f"{api_key_env_var} is required for structured extraction "
                f"provider {binding.provider!r}",
                retryable=False,
            )

        base_url = _string_model_parameter(
            binding.parameters,
            key="base_url",
            default="https://api.openai.com/v1",
        ).rstrip("/")
        endpoint_path = _endpoint_path(
            _string_model_parameter(
                binding.parameters,
                key="endpoint_path",
                default="/chat/completions",
            )
        )
        timeout_seconds = _positive_float_model_parameter(
            binding.parameters,
            key="timeout_seconds",
            default=60.0,
        )
        schema_name = _string_model_parameter(
            binding.parameters,
            key="schema_name",
            default="notarius_extraction_result",
        )
        strict_schema = _bool_model_parameter(
            binding.parameters,
            key="strict_schema",
            default=True,
        )
        request_payload = {
            "model": binding.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON that matches the provided schema.",
                },
                {"role": "user", "content": page.rendered_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": page.schema,
                    "strict": strict_schema,
                },
            },
            "temperature": _number_model_parameter(
                binding.parameters,
                key="temperature",
                default=0.0,
            ),
        }
        max_tokens = _optional_positive_int_model_parameter(
            binding.parameters,
            key="max_tokens",
        )
        if max_tokens is not None:
            request_payload["max_tokens"] = max_tokens
        top_p = _optional_number_model_parameter(binding.parameters, key="top_p")
        if top_p is not None:
            request_payload["top_p"] = top_p

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            if self.http_client is None:
                with httpx.Client(timeout=timeout_seconds) as client:
                    response = client.post(
                        f"{base_url}{endpoint_path}",
                        json=request_payload,
                        headers=headers,
                    )
            else:
                response = self.http_client.post(
                    f"{base_url}{endpoint_path}",
                    json=request_payload,
                    headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            response_excerpt = _response_text_excerpt(exc.response)
            raise StructuredExtractionEngineError(
                "OpenAI-compatible structured extraction failed with status "
                f"{status_code} for provider {binding.provider!r}, "
                f"model {binding.model!r}, page {page.page_number}: "
                f"{response_excerpt}",
                retryable=_retryable_provider_status(status_code),
            ) from exc
        except httpx.TimeoutException as exc:
            raise StructuredExtractionEngineError(
                "OpenAI-compatible structured extraction timed out for provider "
                f"{binding.provider!r}, model {binding.model!r}, "
                f"page {page.page_number}",
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise StructuredExtractionEngineError(
                "OpenAI-compatible structured extraction transport failed for "
                f"provider {binding.provider!r}, model {binding.model!r}, "
                f"page {page.page_number}: {exc.__class__.__name__}: {exc}",
                retryable=True,
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise StructuredExtractionEngineError(
                "OpenAI-compatible structured extraction response was not valid "
                f"JSON for provider {binding.provider!r}, model {binding.model!r}, "
                f"page {page.page_number}",
                retryable=False,
            ) from exc
        return _openai_compatible_structured_output(
            payload,
            provider=binding.provider,
            model=binding.model,
            page_number=page.page_number,
        )


class LocalTextOcrEngine:
    engine_id = "local.text"

    def extract_page(self, page: OcrPageInput) -> OcrPageResultPayload:
        raw_metadata_text = page.image_artifact.metadata.get("ocr_text")
        if isinstance(raw_metadata_text, str):
            text = raw_metadata_text
            text_source = "artifact.metadata.ocr_text"
        else:
            text = page.payload.decode("utf-8", errors="replace")
            text_source = "utf8_payload"

        return OcrPageResultPayload(
            page_number=page.page_number,
            engine=self.engine_id,
            text=text,
            confidence=1.0 if text_source == "artifact.metadata.ocr_text" else None,
            image_artifact_id=str(page.image_artifact.id),
            runtime={
                "byte_size": len(page.payload),
                "text_source": text_source,
                "language_hints": list(page.language_hints),
            },
        )


class TesseractOcrEngine:
    engine_id = "local.tesseract"

    def extract_page(self, page: OcrPageInput) -> OcrPageResultPayload:
        language = _tesseract_language(page.language_hints, page.engine_config)
        psm_mode = _positive_int_engine_config(
            page.engine_config,
            key="psm",
            default=6,
        )
        oem_mode = _positive_int_engine_config(
            page.engine_config,
            key="oem",
            default=3,
        )
        image = _load_pil_image(page.payload)
        tesseract_config = f"--psm {psm_mode} --oem {oem_mode}"
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            lang=language,
            config=tesseract_config,
        )
        tokens: list[dict[str, object]] = []
        confidences: list[float] = []
        image_width, image_height = image.size
        for index, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text).strip()
            confidence = _tesseract_confidence(data.get("conf", []), index)
            if text == "" or confidence < 0:
                continue
            left = _tesseract_int(data.get("left", []), index)
            top = _tesseract_int(data.get("top", []), index)
            width = _tesseract_int(data.get("width", []), index)
            height = _tesseract_int(data.get("height", []), index)
            token = {
                "text": text,
                "confidence": confidence,
                "bbox": [left, top, left + width, top + height],
                "normalized_bbox": _normalized_bbox(
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    image_width=image_width,
                    image_height=image_height,
                ),
            }
            tokens.append(token)
            confidences.append(confidence)

        return OcrPageResultPayload(
            page_number=page.page_number,
            engine=self.engine_id,
            text=" ".join(str(token["text"]) for token in tokens),
            tokens=tokens,
            confidence=sum(confidences) / len(confidences) if confidences else None,
            image_artifact_id=str(page.image_artifact.id),
            runtime={
                "byte_size": len(page.payload),
                "language": language,
                "psm": psm_mode,
                "oem": oem_mode,
                "image_width": image_width,
                "image_height": image_height,
            },
        )


class MistralOcrEngine:
    engine_id = "mistral.ocr"

    def __init__(self, http_client: httpx.Client | None = None) -> None:
        self.http_client = http_client

    def extract_page(self, page: OcrPageInput) -> OcrPageResultPayload:
        api_key_env_var = _string_engine_config(
            page.engine_config,
            key="api_key_env_var",
            default="MISTRAL_API_KEY",
        )
        api_key = os.getenv(api_key_env_var)
        if api_key is None or api_key == "":
            raise ValueError(f"{api_key_env_var} is required for Mistral OCR")

        model = _string_engine_config(
            page.engine_config,
            key="model",
            default="mistral-ocr-latest",
        )
        base_url = _string_engine_config(
            page.engine_config,
            key="base_url",
            default="https://api.mistral.ai",
        ).rstrip("/")
        timeout_seconds = _positive_float_engine_config(
            page.engine_config,
            key="timeout_seconds",
            default=60.0,
        )
        include_blocks = _string_list_engine_config(
            page.engine_config,
            key="include_blocks",
            default=("text",),
        )
        image_media_type = _image_media_type(page.image_artifact)
        document_url = (
            f"data:{image_media_type};base64,"
            f"{base64.b64encode(page.payload).decode('ascii')}"
        )
        request_payload = {
            "model": model,
            "document": {
                "type": "image_url",
                "image_url": document_url,
            },
            "include_blocks": include_blocks,
            "bbox_annotation_format": None,
            "document_annotation_format": None,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            if self.http_client is None:
                with httpx.Client(timeout=timeout_seconds) as client:
                    response = client.post(
                        f"{base_url}/v1/ocr",
                        json=request_payload,
                        headers=headers,
                    )
            else:
                response = self.http_client.post(
                    f"{base_url}/v1/ocr",
                    json=request_payload,
                    headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            response_excerpt = _response_text_excerpt(exc.response)
            raise OcrPageEngineError(
                "Mistral OCR request failed with status "
                f"{status_code} for model {model!r} at {base_url!r}: "
                f"{response_excerpt}",
                retryable=_retryable_mistral_status(status_code),
            ) from exc
        except httpx.TimeoutException as exc:
            raise OcrPageEngineError(
                "Mistral OCR request timed out for model "
                f"{model!r} at {base_url!r}",
                retryable=True,
            ) from exc
        except httpx.TransportError as exc:
            raise OcrPageEngineError(
                "Mistral OCR transport failed for model "
                f"{model!r} at {base_url!r}: "
                f"{exc.__class__.__name__}: {exc}",
                retryable=True,
            ) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ValueError(
                "Mistral OCR response was not valid JSON"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("Mistral OCR response JSON was not an object")
        pages = payload.get("pages")
        if not isinstance(pages, list) or not pages:
            raise ValueError("Mistral OCR response did not include pages")
        first_page = pages[0]
        if not isinstance(first_page, dict):
            raise ValueError("Mistral OCR response page is not an object")
        text = _mistral_page_text(first_page)
        blocks = _mistral_page_blocks(first_page)
        confidence = _mistral_confidence(first_page)

        return OcrPageResultPayload(
            page_number=page.page_number,
            engine=self.engine_id,
            text=text,
            blocks=blocks,
            confidence=confidence,
            image_artifact_id=str(page.image_artifact.id),
            runtime={
                "byte_size": len(page.payload),
                "model": model,
                "base_url": base_url,
                "include_blocks": include_blocks,
                "provider_page_count": len(pages),
            },
        )


def _load_pil_image(payload: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(payload))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("source page payload is not a readable image") from exc
    return image.convert("RGB")


def _tesseract_language(
    language_hints: tuple[str, ...],
    engine_config: dict[str, object],
) -> str:
    raw_language = engine_config.get("language")
    if isinstance(raw_language, str) and raw_language != "":
        return raw_language
    if language_hints:
        return "+".join(language_hints)
    return "eng"


def _string_engine_config(
    engine_config: dict[str, object],
    *,
    key: str,
    default: str,
) -> str:
    value = engine_config.get(key, default)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"engine_config.{key} must be a non-empty string")
    return value


def _string_list_engine_config(
    engine_config: dict[str, object],
    *,
    key: str,
    default: tuple[str, ...],
) -> list[str]:
    value = engine_config.get(key, list(default))
    if not isinstance(value, list) or any(
        not isinstance(item, str) or item == "" for item in value
    ):
        raise ValueError(f"engine_config.{key} must be a list of non-empty strings")
    return list(value)


def _positive_int_engine_config(
    engine_config: dict[str, object],
    *,
    key: str,
    default: int,
) -> int:
    value = engine_config.get(key, default)
    if type(value) is not int or value <= 0:
        raise ValueError(f"engine_config.{key} must be a positive integer")
    return value


def _positive_float_engine_config(
    engine_config: dict[str, object],
    *,
    key: str,
    default: float,
) -> float:
    value = engine_config.get(key, default)
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"engine_config.{key} must be a positive number")
    return float(value)


def _string_model_parameter(
    parameters: dict[str, object],
    *,
    key: str,
    default: str,
) -> str:
    value = parameters.get(key, default)
    if not isinstance(value, str) or value == "":
        raise StructuredExtractionEngineError(
            f"model.parameters.{key} must be a non-empty string",
            retryable=False,
        )
    return value


def _endpoint_path(value: str) -> str:
    return value if value.startswith("/") else f"/{value}"


def _bool_model_parameter(
    parameters: dict[str, object],
    *,
    key: str,
    default: bool,
) -> bool:
    value = parameters.get(key, default)
    if type(value) is not bool:
        raise StructuredExtractionEngineError(
            f"model.parameters.{key} must be a boolean",
            retryable=False,
        )
    return value


def _number_model_parameter(
    parameters: dict[str, object],
    *,
    key: str,
    default: float,
) -> float:
    value = parameters.get(key, default)
    if not isinstance(value, int | float):
        raise StructuredExtractionEngineError(
            f"model.parameters.{key} must be a number",
            retryable=False,
        )
    return float(value)


def _optional_number_model_parameter(
    parameters: dict[str, object],
    *,
    key: str,
) -> float | None:
    value = parameters.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise StructuredExtractionEngineError(
            f"model.parameters.{key} must be a number when provided",
            retryable=False,
        )
    return float(value)


def _positive_float_model_parameter(
    parameters: dict[str, object],
    *,
    key: str,
    default: float,
) -> float:
    value = parameters.get(key, default)
    if not isinstance(value, int | float) or value <= 0:
        raise StructuredExtractionEngineError(
            f"model.parameters.{key} must be a positive number",
            retryable=False,
        )
    return float(value)


def _optional_positive_int_model_parameter(
    parameters: dict[str, object],
    *,
    key: str,
) -> int | None:
    value = parameters.get(key)
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise StructuredExtractionEngineError(
            f"model.parameters.{key} must be a positive integer when provided",
            retryable=False,
        )
    return value


def _retryable_mistral_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _retryable_provider_status(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}


def _response_text_excerpt(response: httpx.Response, limit: int = 500) -> str:
    text = response.text.strip()
    if text == "":
        return "<empty response body>"
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _openai_compatible_structured_output(
    payload: object,
    *,
    provider: str,
    model: str,
    page_number: int,
) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise StructuredExtractionEngineError(
            "OpenAI-compatible structured extraction response JSON was not an "
            f"object for provider {provider!r}, model {model!r}, page {page_number}",
            retryable=False,
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise StructuredExtractionEngineError(
            "OpenAI-compatible structured extraction response did not include "
            f"choices for provider {provider!r}, model {model!r}, page {page_number}",
            retryable=False,
        )
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise StructuredExtractionEngineError(
            "OpenAI-compatible structured extraction first choice was not an "
            f"object for provider {provider!r}, model {model!r}, page {page_number}",
            retryable=False,
        )
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise StructuredExtractionEngineError(
            "OpenAI-compatible structured extraction choice did not include a "
            f"message object for provider {provider!r}, model {model!r}, "
            f"page {page_number}",
            retryable=False,
        )

    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return parsed

    content = message.get("content")
    if isinstance(content, dict):
        return content
    if not isinstance(content, str) or content == "":
        raise StructuredExtractionEngineError(
            "OpenAI-compatible structured extraction message content was empty "
            f"for provider {provider!r}, model {model!r}, page {page_number}",
            retryable=False,
        )
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError as exc:
        raise StructuredExtractionEngineError(
            "OpenAI-compatible structured extraction message content was not "
            f"valid JSON for provider {provider!r}, model {model!r}, "
            f"page {page_number}: {exc.msg}",
            retryable=False,
        ) from exc
    if not isinstance(decoded, dict):
        raise StructuredExtractionEngineError(
            "OpenAI-compatible structured extraction message content did not "
            f"decode to an object for provider {provider!r}, model {model!r}, "
            f"page {page_number}",
            retryable=False,
        )
    return decoded


def _tesseract_confidence(values: object, index: int) -> float:
    if not isinstance(values, list) or index >= len(values):
        return -1.0
    try:
        return float(values[index])
    except (TypeError, ValueError):
        return -1.0


def _tesseract_int(values: object, index: int) -> int:
    if not isinstance(values, list) or index >= len(values):
        return 0
    try:
        return int(float(values[index]))
    except (TypeError, ValueError):
        return 0


def _normalized_bbox(
    *,
    left: int,
    top: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> list[int]:
    if image_width <= 0 or image_height <= 0:
        return [0, 0, 0, 0]
    return [
        int(1000 * left / image_width),
        int(1000 * top / image_height),
        int(1000 * (left + width) / image_width),
        int(1000 * (top + height) / image_height),
    ]


def _image_media_type(artifact: Artifact) -> str:
    raw_content_type = artifact.metadata.get("content_type")
    if isinstance(raw_content_type, str) and raw_content_type.startswith("image/"):
        return raw_content_type
    return "image/png"


def _mistral_page_text(page: dict[str, object]) -> str:
    raw_markdown = page.get("markdown")
    if isinstance(raw_markdown, str):
        return raw_markdown
    raw_text = page.get("text")
    if isinstance(raw_text, str):
        return raw_text
    return ""


def _mistral_page_blocks(page: dict[str, object]) -> list[dict[str, object]]:
    raw_blocks = page.get("blocks")
    if not isinstance(raw_blocks, list):
        return []
    return [block for block in raw_blocks if isinstance(block, dict)]


def _mistral_confidence(page: dict[str, object]) -> float | None:
    raw_confidence = page.get("confidence")
    if isinstance(raw_confidence, int | float):
        return float(raw_confidence)
    dimensions = page.get("dimensions")
    if isinstance(dimensions, dict):
        raw_dimension_confidence = dimensions.get("confidence")
        if isinstance(raw_dimension_confidence, int | float):
            return float(raw_dimension_confidence)
    return None


@dataclass(frozen=True, slots=True)
class OcrExecutionConfig:
    engine: OcrPageEnginePort
    language_hints: tuple[str, ...]
    engine_config: dict[str, object]


@dataclass(frozen=True, slots=True)
class OcrPageArtifacts:
    page_result_artifact: Artifact
    page_result_payload: OcrPageResultPayload
    request_trace_artifact: Artifact
    response_trace_artifact: Artifact


def _default_ocr_engines() -> dict[str, OcrPageEnginePort]:
    return {
        LocalTextOcrEngine.engine_id: LocalTextOcrEngine(),
        MistralOcrEngine.engine_id: MistralOcrEngine(),
        TesseractOcrEngine.engine_id: TesseractOcrEngine(),
    }


def _ocr_execution_config(
    raw_config: object,
    engines: Mapping[str, OcrPageEnginePort],
    operator_id: str,
) -> OcrExecutionConfig:
    if not isinstance(raw_config, dict):
        raise NodeRunExecutionError(
            f"{operator_id} expected workflow_node_config metadata to be an object",
            retryable=False,
        )

    raw_engine = raw_config.get("engine", LocalTextOcrEngine.engine_id)
    if not isinstance(raw_engine, str) or raw_engine == "":
        raise NodeRunExecutionError(
            f"{operator_id} requires workflow_node_config.engine to be a "
            "non-empty string",
            retryable=False,
        )
    engine = engines.get(raw_engine)
    if engine is None:
        raise NodeRunExecutionError(
            f"{operator_id} has no OCR engine registered for {raw_engine!r}",
            retryable=False,
        )

    raw_language_hints = raw_config.get("language_hints", [])
    if not isinstance(raw_language_hints, list) or any(
        not isinstance(language_hint, str) or language_hint == ""
        for language_hint in raw_language_hints
    ):
        raise NodeRunExecutionError(
            f"{operator_id} requires workflow_node_config.language_hints "
            "to be a list of non-empty strings",
            retryable=False,
        )
    raw_engine_config = raw_config.get("engine_config", {})
    if not isinstance(raw_engine_config, dict):
        raise NodeRunExecutionError(
            f"{operator_id} requires workflow_node_config.engine_config "
            "to be an object",
            retryable=False,
        )
    return OcrExecutionConfig(
        engine=engine,
        language_hints=tuple(raw_language_hints),
        engine_config=dict(raw_engine_config),
    )


class OcrExtractPagesHandler:
    def __init__(
        self,
        payload_storage: ArtifactPayloadStoragePort,
        engines: Mapping[str, OcrPageEnginePort] | None = None,
    ) -> None:
        self.payload_storage = payload_storage
        self.engines = dict(engines or _default_ocr_engines())

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        ocr_config = _ocr_execution_config(
            request.node_run.metadata.get("workflow_node_config", {}),
            self.engines,
            OCR_EXTRACT_PAGES_OPERATOR_ID,
        )
        engine = ocr_config.engine
        language_hints = ocr_config.language_hints
        engine_config = ocr_config.engine_config

        pages = request.input_artifacts.get("pages")
        if not isinstance(pages, ArtifactSequenceInput):
            raise NodeRunExecutionError(
                "ocr.extract_pages requires pages to be an artifact sequence input",
                retryable=False,
            )

        page_result_artifacts: list[Artifact] = []
        page_result_payloads: list[OcrPageResultPayload] = []
        request_trace_artifacts: list[Artifact] = []
        response_trace_artifacts: list[Artifact] = []
        for sequence_index, image_artifact in enumerate(pages.artifacts, start=1):
            page_number = _page_number(image_artifact, sequence_index)
            image_payload = _load_artifact_payload(
                self.payload_storage,
                image_artifact,
                port_name="pages",
            )
            request_trace_payload = OcrRequestTracePayload(
                sequence_index=sequence_index,
                page_number=page_number,
                engine=engine.engine_id,
                image_artifact_id=str(image_artifact.id),
                input_sequence_id=str(pages.sequence.id),
                image_payload_ref=image_artifact.payload_ref,
                image_media_type=_image_media_type(image_artifact),
                payload_byte_size=len(image_payload),
                language_hints=list(language_hints),
                engine_config=_redacted_config(engine_config),
            )
            try:
                result_payload = engine.extract_page(
                    OcrPageInput(
                        page_number=page_number,
                        image_artifact=image_artifact,
                        payload=image_payload,
                        language_hints=language_hints,
                        engine_config=engine_config,
                    )
                )
            except NodeRunExecutionError:
                raise
            except OcrPageEngineError as exc:
                raise NodeRunExecutionError(
                    "OCR engine failed for page "
                    f"{page_number} and artifact {image_artifact.id}: {exc}",
                    retryable=exc.retryable,
                ) from exc
            except Exception as exc:
                raise NodeRunExecutionError(
                    "OCR engine failed for page "
                    f"{page_number} and artifact {image_artifact.id}: {exc}",
                    retryable=False,
                ) from exc
            result_bytes = result_payload.model_dump_json(indent=2).encode("utf-8")
            key = (
                f"workflow-runs/{request.node_run.workflow_run_id}/"
                f"node-runs/{request.node_run.id}/ocr-pages/"
                f"{page_number:04d}.json"
            )
            stored = self.payload_storage.save(
                SaveArtifactPayloadCommand(
                    bucket="ocr-page-results",
                    key=key,
                    payload=result_bytes,
                    overwrite=True,
                )
            )
            artifact_metadata = {
                "page_number": page_number,
                "source_image_artifact_id": str(image_artifact.id),
                "engine": result_payload.engine,
                "text_length": len(result_payload.text),
                "content_type": "application/json",
                "byte_size": stored.byte_size,
            }
            raw_model = result_payload.runtime.get("model")
            if isinstance(raw_model, str) and raw_model != "":
                artifact_metadata["model"] = raw_model

            page_result_artifact = Artifact(
                artifact_type="ocr.page_result",
                schema_version=1,
                workflow_run_id=request.node_run.workflow_run_id,
                producer_node_run_id=request.node_run.id,
                payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
                producer_operator_id=request.node_run.operator_id,
                producer_operator_version=request.node_run.operator_version,
                input_artifact_ids=[image_artifact.id],
                content_hash=stored.sha256,
                metadata=artifact_metadata,
            )
            request_trace_artifact = _save_json_artifact(
                self.payload_storage,
                request,
                artifact_type="ocr.request_trace",
                bucket="ocr-request-traces",
                filename=f"ocr-request-traces/{page_number:04d}.json",
                payload=request_trace_payload,
                metadata={
                    "sequence_index": sequence_index,
                    "page_number": page_number,
                    "engine": engine.engine_id,
                    "source_image_artifact_id": str(image_artifact.id),
                },
                input_artifact_ids=[image_artifact.id],
            )
            response_trace_artifact = _save_json_artifact(
                self.payload_storage,
                request,
                artifact_type="ocr.response_trace",
                bucket="ocr-response-traces",
                filename=f"ocr-response-traces/{page_number:04d}.json",
                payload=OcrResponseTracePayload(
                    sequence_index=sequence_index,
                    page_number=page_number,
                    engine=result_payload.engine,
                    image_artifact_id=str(image_artifact.id),
                    ocr_result_artifact_id=str(page_result_artifact.id),
                    text_length=len(result_payload.text),
                    block_count=len(result_payload.blocks),
                    token_count=len(result_payload.tokens),
                    confidence=result_payload.confidence,
                    runtime=result_payload.runtime,
                ),
                metadata={
                    "sequence_index": sequence_index,
                    "page_number": page_number,
                    "engine": result_payload.engine,
                    "source_image_artifact_id": str(image_artifact.id),
                    "ocr_result_artifact_id": str(page_result_artifact.id),
                },
                input_artifact_ids=[
                    image_artifact.id,
                    page_result_artifact.id,
                    request_trace_artifact.id,
                ],
            )
            page_result_artifacts.append(page_result_artifact)
            page_result_payloads.append(result_payload)
            request_trace_artifacts.append(request_trace_artifact)
            response_trace_artifacts.append(response_trace_artifact)

        sequence = ArtifactSequence(
            artifact_type="ocr.page_result",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in page_result_artifacts],
            index_key=pages.sequence.index_key,
            metadata={
                "engine": engine.engine_id,
                "input_sequence_id": str(pages.sequence.id),
                "page_count": len(page_result_artifacts),
            },
        )
        request_trace_sequence = ArtifactSequence(
            artifact_type="ocr.request_trace",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in request_trace_artifacts],
            index_key=pages.sequence.index_key,
            metadata={
                "engine": engine.engine_id,
                "input_sequence_id": str(pages.sequence.id),
                "page_count": len(request_trace_artifacts),
            },
        )
        response_trace_sequence = ArtifactSequence(
            artifact_type="ocr.response_trace",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in response_trace_artifacts],
            index_key=pages.sequence.index_key,
            metadata={
                "engine": engine.engine_id,
                "input_sequence_id": str(pages.sequence.id),
                "page_count": len(response_trace_artifacts),
            },
        )
        document_payload = OcrDocumentResultPayload(
            engine=engine.engine_id,
            page_count=len(page_result_artifacts),
            text="\n\n".join(payload.text for payload in page_result_payloads),
            page_result_artifact_ids=[
                str(artifact.id) for artifact in page_result_artifacts
            ],
            request_trace_sequence_id=str(request_trace_sequence.id),
            response_trace_sequence_id=str(response_trace_sequence.id),
            source_page_sequence_id=str(pages.sequence.id),
            language_hints=list(language_hints),
            runtime={"engine_config": _redacted_config(engine_config)},
        )
        document_artifact = _save_json_artifact(
            self.payload_storage,
            request,
            artifact_type="ocr.document_result",
            bucket="ocr-document-results",
            filename="ocr-document-result.json",
            payload=document_payload,
            metadata={
                "engine": engine.engine_id,
                "page_count": len(page_result_artifacts),
                "text_length": len(document_payload.text),
                "source_page_sequence_id": str(pages.sequence.id),
                "ocr_page_sequence_id": str(sequence.id),
                "request_trace_sequence_id": str(request_trace_sequence.id),
                "response_trace_sequence_id": str(response_trace_sequence.id),
            },
            input_artifact_ids=[artifact.id for artifact in page_result_artifacts],
        )
        invocation_traces = [
            InvocationTrace(
                node_run_id=request.node_run.id,
                invocation_type=OCR_EXTRACT_PAGES_OPERATOR_ID,
                input_artifact_refs=[
                    pages.artifacts[index].ref(),
                    request_trace_artifact.ref(),
                ],
                output_artifact_refs=[
                    page_result_artifacts[index].ref(),
                    response_trace_artifact.ref(),
                ],
                provider=_ocr_provider(engine.engine_id),
                model=_ocr_model(engine.engine_id, [page_result_artifacts[index]]),
                request_ref=request_trace_artifact.payload_ref,
                response_ref=response_trace_artifact.payload_ref,
                runtime={
                    "page_count": len(page_result_artifacts),
                    "sequence_index": index + 1,
                    "page_number": _page_number(pages.artifacts[index], index + 1),
                    "engine": engine.engine_id,
                    "language_hints": list(language_hints),
                },
                metadata={
                    "output_sequence_id": str(sequence.id),
                    "request_trace_sequence_id": str(request_trace_sequence.id),
                    "response_trace_sequence_id": str(response_trace_sequence.id),
                    "request_trace_artifact_id": str(request_trace_artifact.id),
                    "response_trace_artifact_id": str(response_trace_artifact.id),
                    "document_artifact_id": str(document_artifact.id),
                },
            )
            for index, (request_trace_artifact, response_trace_artifact) in enumerate(
                zip(
                    request_trace_artifacts,
                    response_trace_artifacts,
                    strict=True,
                )
            )
        ]
        return NodeExecutionResult(
            output_artifact_refs={
                "ocr_pages": sequence.ref(),
                "ocr_document": document_artifact.ref(),
            },
            artifacts=[
                *page_result_artifacts,
                *request_trace_artifacts,
                *response_trace_artifacts,
                document_artifact,
            ],
            artifact_sequences=[
                sequence,
                request_trace_sequence,
                response_trace_sequence,
            ],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs={
                        "pages": [artifact.ref() for artifact in pages.artifacts],
                    },
                    policies={"engine": engine.engine_id},
                    metadata={"input_sequence_id": str(pages.sequence.id)},
                )
            ],
            invocation_traces=invocation_traces,
        )


class OcrExtractPageHandler:
    def __init__(
        self,
        payload_storage: ArtifactPayloadStoragePort,
        engines: Mapping[str, OcrPageEnginePort] | None = None,
    ) -> None:
        self.payload_storage = payload_storage
        self.engines = dict(engines or _default_ocr_engines())

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        ocr_config = _ocr_execution_config(
            request.node_run.metadata.get("workflow_node_config", {}),
            self.engines,
            OCR_EXTRACT_PAGE_OPERATOR_ID,
        )
        image_artifact = _single_ocr_image_artifact(request)
        sequence_index = _node_run_positive_int_metadata(
            request,
            "map_item_index",
            fallback=1,
        )
        source_sequence_id = _node_run_string_metadata(
            request,
            "map_source_sequence_id",
        )
        page_artifacts = _extract_ocr_page_artifacts(
            self.payload_storage,
            request,
            image_artifact=image_artifact,
            sequence_index=sequence_index,
            input_sequence_id=source_sequence_id,
            ocr_config=ocr_config,
        )
        page_artifacts.page_result_artifact.metadata["request_trace_artifact_ref"] = (
            _artifact_ref_metadata(page_artifacts.request_trace_artifact.ref())
        )
        page_artifacts.page_result_artifact.metadata["response_trace_artifact_ref"] = (
            _artifact_ref_metadata(page_artifacts.response_trace_artifact.ref())
        )
        return NodeExecutionResult(
            output_artifact_refs={
                "ocr_pages": [page_artifacts.page_result_artifact.ref()]
            },
            artifacts=[
                page_artifacts.page_result_artifact,
                page_artifacts.request_trace_artifact,
                page_artifacts.response_trace_artifact,
            ],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs={"pages": [image_artifact.ref()]},
                    policies={"engine": ocr_config.engine.engine_id},
                    metadata={
                        "source_page_sequence_id": source_sequence_id,
                        "sequence_index": sequence_index,
                    },
                )
            ],
            invocation_traces=[
                InvocationTrace(
                    node_run_id=request.node_run.id,
                    invocation_type=OCR_EXTRACT_PAGE_OPERATOR_ID,
                    input_artifact_refs=[
                        image_artifact.ref(),
                        page_artifacts.request_trace_artifact.ref(),
                    ],
                    output_artifact_refs=[
                        page_artifacts.page_result_artifact.ref(),
                        page_artifacts.response_trace_artifact.ref(),
                    ],
                    provider=_ocr_provider(ocr_config.engine.engine_id),
                    model=_ocr_model(
                        ocr_config.engine.engine_id,
                        [page_artifacts.page_result_artifact],
                    ),
                    request_ref=page_artifacts.request_trace_artifact.payload_ref,
                    response_ref=page_artifacts.response_trace_artifact.payload_ref,
                    runtime={
                        "sequence_index": sequence_index,
                        "page_number": page_artifacts.page_result_payload.page_number,
                        "engine": ocr_config.engine.engine_id,
                        "language_hints": list(ocr_config.language_hints),
                    },
                    metadata={
                        "source_page_sequence_id": source_sequence_id,
                        "request_trace_artifact_id": str(
                            page_artifacts.request_trace_artifact.id
                        ),
                        "response_trace_artifact_id": str(
                            page_artifacts.response_trace_artifact.id
                        ),
                    },
                )
            ],
        )


class OcrCollectPagesHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        ocr_pages = request.input_artifacts.get("ocr_pages")
        if not isinstance(ocr_pages, ArtifactSequenceInput):
            raise NodeRunExecutionError(
                "ocr.collect_pages requires ocr_pages to be an artifact sequence input",
                retryable=False,
            )
        if not ocr_pages.artifacts:
            raise NodeRunExecutionError(
                "ocr.collect_pages requires at least one OCR page-result artifact",
                retryable=False,
            )

        page_payloads = [
            _load_ocr_page_result_payload(
                self.payload_storage,
                artifact,
                "ocr_pages",
            )
            for artifact in ocr_pages.artifacts
        ]
        engine = page_payloads[0].engine
        if any(payload.engine != engine for payload in page_payloads):
            raise NodeRunExecutionError(
                "ocr.collect_pages cannot collect OCR page results from mixed engines",
                retryable=False,
            )
        request_trace_refs = [
            trace_ref
            for artifact in ocr_pages.artifacts
            if (
                trace_ref := _metadata_artifact_ref(
                    artifact,
                    "request_trace_artifact_ref",
                )
            )
            is not None
        ]
        response_trace_refs = [
            trace_ref
            for artifact in ocr_pages.artifacts
            if (
                trace_ref := _metadata_artifact_ref(
                    artifact,
                    "response_trace_artifact_ref",
                )
            )
            is not None
        ]
        request_trace_sequence = ArtifactSequence(
            artifact_type="ocr.request_trace",
            schema_version=1,
            item_refs=request_trace_refs,
            index_key=ocr_pages.sequence.index_key,
            metadata={
                "engine": engine,
                "ocr_page_sequence_id": str(ocr_pages.sequence.id),
                "page_count": len(request_trace_refs),
            },
        )
        response_trace_sequence = ArtifactSequence(
            artifact_type="ocr.response_trace",
            schema_version=1,
            item_refs=response_trace_refs,
            index_key=ocr_pages.sequence.index_key,
            metadata={
                "engine": engine,
                "ocr_page_sequence_id": str(ocr_pages.sequence.id),
                "page_count": len(response_trace_refs),
            },
        )
        document_payload = OcrDocumentResultPayload(
            engine=engine,
            page_count=len(ocr_pages.artifacts),
            text="\n\n".join(payload.text for payload in page_payloads),
            page_result_artifact_ids=[
                str(artifact.id) for artifact in ocr_pages.artifacts
            ],
            request_trace_sequence_id=str(request_trace_sequence.id),
            response_trace_sequence_id=str(response_trace_sequence.id),
            source_page_sequence_id=_source_page_sequence_id(ocr_pages.artifacts),
            language_hints=_runtime_string_list(page_payloads[0], "language_hints"),
            runtime={
                "engine_config": _runtime_object(
                    page_payloads[0],
                    "engine_config",
                )
            },
        )
        document_artifact = _save_json_artifact(
            self.payload_storage,
            request,
            artifact_type="ocr.document_result",
            bucket="ocr-document-results",
            filename="ocr-document-result.json",
            payload=document_payload,
            metadata={
                "engine": engine,
                "page_count": len(ocr_pages.artifacts),
                "text_length": len(document_payload.text),
                "source_page_sequence_id": document_payload.source_page_sequence_id,
                "ocr_page_sequence_id": str(ocr_pages.sequence.id),
                "request_trace_sequence_id": str(request_trace_sequence.id),
                "response_trace_sequence_id": str(response_trace_sequence.id),
            },
            input_artifact_ids=[artifact.id for artifact in ocr_pages.artifacts],
        )
        return NodeExecutionResult(
            output_artifact_refs={
                "ocr_pages": ocr_pages.sequence.ref(),
                "ocr_document": document_artifact.ref(),
            },
            artifacts=[document_artifact],
            artifact_sequences=[
                request_trace_sequence,
                response_trace_sequence,
            ],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs={
                        "ocr_pages": [
                            artifact.ref() for artifact in ocr_pages.artifacts
                        ],
                    },
                    metadata={"ocr_page_sequence_id": str(ocr_pages.sequence.id)},
                )
            ],
            invocation_traces=[
                InvocationTrace(
                    node_run_id=request.node_run.id,
                    invocation_type=OCR_COLLECT_PAGES_OPERATOR_ID,
                    input_artifact_refs=[
                        artifact.ref() for artifact in ocr_pages.artifacts
                    ],
                    output_artifact_refs=[document_artifact.ref()],
                    provider=_ocr_provider(engine),
                    model=_ocr_model(engine, ocr_pages.artifacts),
                    runtime={
                        "page_count": len(ocr_pages.artifacts),
                        "engine": engine,
                    },
                    metadata={
                        "ocr_page_sequence_id": str(ocr_pages.sequence.id),
                        "request_trace_sequence_id": str(request_trace_sequence.id),
                        "response_trace_sequence_id": str(response_trace_sequence.id),
                    },
                )
            ],
        )


def _single_ocr_image_artifact(request: NodeExecutionRequest) -> Artifact:
    pages = request.input_artifacts.get("pages")
    if isinstance(pages, ArtifactSequenceInput):
        artifacts = pages.artifacts
    elif isinstance(pages, list) and all(isinstance(item, Artifact) for item in pages):
        artifacts = pages
    else:
        raise NodeRunExecutionError(
            "ocr.extract_page requires pages to contain one source.page_image artifact",
            retryable=False,
        )
    if len(artifacts) != 1:
        raise NodeRunExecutionError(
            "ocr.extract_page requires exactly one source.page_image artifact",
            retryable=False,
        )
    return artifacts[0]


def _node_run_positive_int_metadata(
    request: NodeExecutionRequest,
    field_name: str,
    *,
    fallback: int,
) -> int:
    raw_value = request.node_run.metadata.get(field_name)
    return raw_value if type(raw_value) is int and raw_value > 0 else fallback


def _node_run_string_metadata(
    request: NodeExecutionRequest,
    field_name: str,
) -> str:
    raw_value = request.node_run.metadata.get(field_name)
    return raw_value if isinstance(raw_value, str) else ""


def _extract_ocr_page_artifacts(
    payload_storage: ArtifactPayloadStoragePort,
    request: NodeExecutionRequest,
    *,
    image_artifact: Artifact,
    sequence_index: int,
    input_sequence_id: str,
    ocr_config: OcrExecutionConfig,
) -> OcrPageArtifacts:
    page_number = _page_number(image_artifact, sequence_index)
    image_payload = _load_artifact_payload(
        payload_storage,
        image_artifact,
        port_name="pages",
    )
    request_trace_payload = OcrRequestTracePayload(
        sequence_index=sequence_index,
        page_number=page_number,
        engine=ocr_config.engine.engine_id,
        image_artifact_id=str(image_artifact.id),
        input_sequence_id=input_sequence_id,
        image_payload_ref=image_artifact.payload_ref,
        image_media_type=_image_media_type(image_artifact),
        payload_byte_size=len(image_payload),
        language_hints=list(ocr_config.language_hints),
        engine_config=_redacted_config(ocr_config.engine_config),
    )
    try:
        result_payload = ocr_config.engine.extract_page(
            OcrPageInput(
                page_number=page_number,
                image_artifact=image_artifact,
                payload=image_payload,
                language_hints=ocr_config.language_hints,
                engine_config=ocr_config.engine_config,
            )
        )
    except NodeRunExecutionError:
        raise
    except OcrPageEngineError as exc:
        raise NodeRunExecutionError(
            "OCR engine failed for page "
            f"{page_number} and artifact {image_artifact.id}: {exc}",
            retryable=exc.retryable,
        ) from exc
    except Exception as exc:
        raise NodeRunExecutionError(
            "OCR engine failed for page "
            f"{page_number} and artifact {image_artifact.id}: {exc}",
            retryable=False,
        ) from exc

    result_payload = result_payload.model_copy(
        update={
            "runtime": {
                **result_payload.runtime,
                "language_hints": list(ocr_config.language_hints),
                "engine_config": _redacted_config(ocr_config.engine_config),
                "source_page_sequence_id": input_sequence_id,
            }
        }
    )
    stored = payload_storage.save(
        SaveArtifactPayloadCommand(
            bucket="ocr-page-results",
            key=(
                f"workflow-runs/{request.node_run.workflow_run_id}/"
                f"node-runs/{request.node_run.id}/ocr-pages/"
                f"{page_number:04d}.json"
            ),
            payload=result_payload.model_dump_json(indent=2).encode("utf-8"),
            overwrite=True,
        )
    )
    artifact_metadata = {
        "sequence_index": sequence_index,
        "page_number": page_number,
        "source_image_artifact_id": str(image_artifact.id),
        "source_page_sequence_id": input_sequence_id,
        "engine": result_payload.engine,
        "text_length": len(result_payload.text),
        "content_type": "application/json",
        "byte_size": stored.byte_size,
    }
    raw_model = result_payload.runtime.get("model")
    if isinstance(raw_model, str) and raw_model != "":
        artifact_metadata["model"] = raw_model

    page_result_artifact = Artifact(
        artifact_type="ocr.page_result",
        schema_version=1,
        workflow_run_id=request.node_run.workflow_run_id,
        producer_node_run_id=request.node_run.id,
        payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
        producer_operator_id=request.node_run.operator_id,
        producer_operator_version=request.node_run.operator_version,
        input_artifact_ids=[image_artifact.id],
        content_hash=stored.sha256,
        metadata=artifact_metadata,
    )
    request_trace_artifact = _save_json_artifact(
        payload_storage,
        request,
        artifact_type="ocr.request_trace",
        bucket="ocr-request-traces",
        filename=f"ocr-request-traces/{page_number:04d}.json",
        payload=request_trace_payload,
        metadata={
            "sequence_index": sequence_index,
            "page_number": page_number,
            "engine": ocr_config.engine.engine_id,
            "source_image_artifact_id": str(image_artifact.id),
        },
        input_artifact_ids=[image_artifact.id],
    )
    response_trace_artifact = _save_json_artifact(
        payload_storage,
        request,
        artifact_type="ocr.response_trace",
        bucket="ocr-response-traces",
        filename=f"ocr-response-traces/{page_number:04d}.json",
        payload=OcrResponseTracePayload(
            sequence_index=sequence_index,
            page_number=page_number,
            engine=result_payload.engine,
            image_artifact_id=str(image_artifact.id),
            ocr_result_artifact_id=str(page_result_artifact.id),
            text_length=len(result_payload.text),
            block_count=len(result_payload.blocks),
            token_count=len(result_payload.tokens),
            confidence=result_payload.confidence,
            runtime=result_payload.runtime,
        ),
        metadata={
            "sequence_index": sequence_index,
            "page_number": page_number,
            "engine": result_payload.engine,
            "source_image_artifact_id": str(image_artifact.id),
            "ocr_result_artifact_id": str(page_result_artifact.id),
        },
        input_artifact_ids=[
            image_artifact.id,
            page_result_artifact.id,
            request_trace_artifact.id,
        ],
    )
    return OcrPageArtifacts(
        page_result_artifact=page_result_artifact,
        page_result_payload=result_payload,
        request_trace_artifact=request_trace_artifact,
        response_trace_artifact=response_trace_artifact,
    )


def _artifact_ref_metadata(artifact_ref: ArtifactRef) -> dict[str, object]:
    return {
        "artifact_id": str(artifact_ref.artifact_id),
        "artifact_type": artifact_ref.artifact_type,
        "schema_version": artifact_ref.schema_version,
        "content_hash": artifact_ref.content_hash,
    }


def _metadata_artifact_ref(
    artifact: Artifact,
    field_name: str,
) -> ArtifactRef | None:
    raw_ref = artifact.metadata.get(field_name)
    if not isinstance(raw_ref, dict):
        return None
    raw_artifact_id = raw_ref.get("artifact_id")
    raw_artifact_type = raw_ref.get("artifact_type")
    raw_schema_version = raw_ref.get("schema_version")
    raw_content_hash = raw_ref.get("content_hash")
    if (
        not isinstance(raw_artifact_id, str)
        or not isinstance(raw_artifact_type, str)
        or type(raw_schema_version) is not int
    ):
        return None
    try:
        artifact_id = UUID(raw_artifact_id)
    except ValueError:
        return None
    return ArtifactRef(
        artifact_id=artifact_id,
        artifact_type=raw_artifact_type,
        schema_version=raw_schema_version,
        content_hash=raw_content_hash if isinstance(raw_content_hash, str) else None,
    )


def _source_page_sequence_id(artifacts: list[Artifact]) -> str:
    raw_value = artifacts[0].metadata.get("source_page_sequence_id")
    return raw_value if isinstance(raw_value, str) else ""


def _runtime_string_list(payload: OcrPageResultPayload, field_name: str) -> list[str]:
    raw_value = payload.runtime.get(field_name)
    if not isinstance(raw_value, list):
        return []
    return [item for item in raw_value if isinstance(item, str)]


def _runtime_object(
    payload: OcrPageResultPayload,
    field_name: str,
) -> dict[str, object]:
    raw_value = payload.runtime.get(field_name)
    if not isinstance(raw_value, dict):
        return {}
    return dict(raw_value)


def _ocr_provider(engine_id: str) -> str:
    if engine_id == MistralOcrEngine.engine_id:
        return "mistral"
    return "local"


def _ocr_model(engine_id: str, artifacts: list[Artifact]) -> str:
    if engine_id != MistralOcrEngine.engine_id or not artifacts:
        return engine_id
    raw_model = artifacts[0].metadata.get("model")
    return raw_model if isinstance(raw_model, str) and raw_model != "" else engine_id


class OcrComparePagesHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = request.node_run.metadata.get("workflow_node_config", {})
        if not isinstance(raw_config, dict):
            raise NodeRunExecutionError(
                "ocr.compare_pages expected workflow_node_config metadata to be an object",
                retryable=False,
            )

        raw_candidate_a_label = raw_config.get("candidate_a_label", "candidate_a")
        if (
            not isinstance(raw_candidate_a_label, str)
            or raw_candidate_a_label == ""
        ):
            raise NodeRunExecutionError(
                "ocr.compare_pages requires workflow_node_config.candidate_a_label "
                "to be a non-empty string",
                retryable=False,
            )
        candidate_a_label = raw_candidate_a_label

        raw_candidate_b_label = raw_config.get("candidate_b_label", "candidate_b")
        if (
            not isinstance(raw_candidate_b_label, str)
            or raw_candidate_b_label == ""
        ):
            raise NodeRunExecutionError(
                "ocr.compare_pages requires workflow_node_config.candidate_b_label "
                "to be a non-empty string",
                retryable=False,
            )
        candidate_b_label = raw_candidate_b_label

        candidate_a_pages = request.input_artifacts.get("candidate_a_pages")
        if not isinstance(candidate_a_pages, ArtifactSequenceInput):
            raise NodeRunExecutionError(
                "ocr.compare_pages requires candidate_a_pages to be an artifact "
                "sequence input",
                retryable=False,
            )
        candidate_b_pages = request.input_artifacts.get("candidate_b_pages")
        if not isinstance(candidate_b_pages, ArtifactSequenceInput):
            raise NodeRunExecutionError(
                "ocr.compare_pages requires candidate_b_pages to be an artifact "
                "sequence input",
                retryable=False,
            )
        if len(candidate_a_pages.artifacts) != len(candidate_b_pages.artifacts):
            raise NodeRunExecutionError(
                "ocr.compare_pages requires sequences with matching page counts: "
                f"candidate_a_pages has {len(candidate_a_pages.artifacts)}, "
                f"candidate_b_pages has {len(candidate_b_pages.artifacts)}",
                retryable=False,
            )

        comparison_artifacts: list[Artifact] = []
        similarity_ratios: list[float] = []
        for sequence_index, artifacts in enumerate(
            zip(candidate_a_pages.artifacts, candidate_b_pages.artifacts, strict=True),
            start=1,
        ):
            candidate_a_artifact, candidate_b_artifact = artifacts
            candidate_a_payload = _load_ocr_page_result_payload(
                self.payload_storage,
                candidate_a_artifact,
                port_name="candidate_a_pages",
            )
            candidate_b_payload = _load_ocr_page_result_payload(
                self.payload_storage,
                candidate_b_artifact,
                port_name="candidate_b_pages",
            )
            similarity_ratio = SequenceMatcher(
                None,
                candidate_a_payload.text,
                candidate_b_payload.text,
                autojunk=False,
            ).ratio()
            similarity_ratios.append(similarity_ratio)
            comparison_payload = OcrComparisonPagePayload(
                sequence_index=sequence_index,
                candidate_a_page_number=candidate_a_payload.page_number,
                candidate_b_page_number=candidate_b_payload.page_number,
                candidate_a_label=candidate_a_label,
                candidate_b_label=candidate_b_label,
                candidate_a_artifact_id=str(candidate_a_artifact.id),
                candidate_b_artifact_id=str(candidate_b_artifact.id),
                candidate_a_engine=candidate_a_payload.engine,
                candidate_b_engine=candidate_b_payload.engine,
                candidate_a_image_artifact_id=candidate_a_payload.image_artifact_id,
                candidate_b_image_artifact_id=candidate_b_payload.image_artifact_id,
                candidate_a_text_length=len(candidate_a_payload.text),
                candidate_b_text_length=len(candidate_b_payload.text),
                similarity_ratio=similarity_ratio,
                equal_text=candidate_a_payload.text == candidate_b_payload.text,
            )
            comparison_bytes = comparison_payload.model_dump_json(indent=2).encode(
                "utf-8"
            )
            key = (
                f"workflow-runs/{request.node_run.workflow_run_id}/"
                f"node-runs/{request.node_run.id}/comparison-pages/"
                f"{sequence_index:04d}.json"
            )
            stored = self.payload_storage.save(
                SaveArtifactPayloadCommand(
                    bucket="ocr-comparison-results",
                    key=key,
                    payload=comparison_bytes,
                    overwrite=True,
                )
            )
            artifact = Artifact(
                artifact_type="ocr.comparison_result",
                schema_version=1,
                workflow_run_id=request.node_run.workflow_run_id,
                producer_node_run_id=request.node_run.id,
                payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
                producer_operator_id=request.node_run.operator_id,
                producer_operator_version=request.node_run.operator_version,
                input_artifact_ids=[candidate_a_artifact.id, candidate_b_artifact.id],
                content_hash=stored.sha256,
                metadata={
                    "sequence_index": sequence_index,
                    "candidate_a_page_number": candidate_a_payload.page_number,
                    "candidate_b_page_number": candidate_b_payload.page_number,
                    "candidate_a_engine": candidate_a_payload.engine,
                    "candidate_b_engine": candidate_b_payload.engine,
                    "similarity_ratio": similarity_ratio,
                    "equal_text": comparison_payload.equal_text,
                    "content_type": "application/json",
                    "byte_size": stored.byte_size,
                },
            )
            comparison_artifacts.append(artifact)

        page_count = len(comparison_artifacts)
        mean_similarity_ratio = (
            sum(similarity_ratios) / page_count if page_count > 0 else None
        )
        metrics_payload = EvaluationMetricsPayload(
            metric_family="ocr_comparison",
            metrics={
                "candidate_a_label": candidate_a_label,
                "candidate_b_label": candidate_b_label,
                "candidate_a_sequence_id": str(candidate_a_pages.sequence.id),
                "candidate_b_sequence_id": str(candidate_b_pages.sequence.id),
                "page_count": page_count,
                "mean_similarity_ratio": mean_similarity_ratio,
                "min_similarity_ratio": (
                    min(similarity_ratios) if similarity_ratios else None
                ),
                "max_similarity_ratio": (
                    max(similarity_ratios) if similarity_ratios else None
                ),
            },
            source_artifact_ids=[
                str(artifact.id)
                for artifact in candidate_a_pages.artifacts + candidate_b_pages.artifacts
            ],
            metadata={
                "candidate_a_sequence_id": str(candidate_a_pages.sequence.id),
                "candidate_b_sequence_id": str(candidate_b_pages.sequence.id),
            },
        )
        metrics_bytes = metrics_payload.model_dump_json(indent=2).encode("utf-8")
        metrics_key = (
            f"workflow-runs/{request.node_run.workflow_run_id}/"
            f"node-runs/{request.node_run.id}/metrics.json"
        )
        stored_metrics = self.payload_storage.save(
            SaveArtifactPayloadCommand(
                bucket="evaluation-metrics",
                key=metrics_key,
                payload=metrics_bytes,
                overwrite=True,
            )
        )
        metrics_artifact = Artifact(
            artifact_type="evaluation.metrics",
            schema_version=1,
            workflow_run_id=request.node_run.workflow_run_id,
            producer_node_run_id=request.node_run.id,
            payload_ref=artifact_payload_ref(
                bucket=stored_metrics.bucket,
                key=stored_metrics.key,
            ),
            producer_operator_id=request.node_run.operator_id,
            producer_operator_version=request.node_run.operator_version,
            input_artifact_ids=[
                artifact.id
                for artifact in candidate_a_pages.artifacts + candidate_b_pages.artifacts
            ],
            content_hash=stored_metrics.sha256,
            metadata={
                "metric_family": "ocr_comparison",
                "page_count": page_count,
                "candidate_a_label": candidate_a_label,
                "candidate_b_label": candidate_b_label,
                "mean_similarity_ratio": mean_similarity_ratio,
                "content_type": "application/json",
                "byte_size": stored_metrics.byte_size,
            },
        )
        sequence = ArtifactSequence(
            artifact_type="ocr.comparison_result",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in comparison_artifacts],
            index_key=candidate_a_pages.sequence.index_key,
            metadata={
                "candidate_a_sequence_id": str(candidate_a_pages.sequence.id),
                "candidate_b_sequence_id": str(candidate_b_pages.sequence.id),
                "page_count": page_count,
                "metrics_artifact_id": str(metrics_artifact.id),
            },
        )

        selected_inputs = {
            "candidate_a_pages": [
                artifact.ref() for artifact in candidate_a_pages.artifacts
            ],
            "candidate_b_pages": [
                artifact.ref() for artifact in candidate_b_pages.artifacts
            ],
        }
        return NodeExecutionResult(
            output_artifact_refs={
                "comparison_pages": sequence.ref(),
                "metrics": metrics_artifact.ref(),
            },
            artifacts=[*comparison_artifacts, metrics_artifact],
            artifact_sequences=[sequence],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs=selected_inputs,
                    policies={
                        "similarity_algorithm": "difflib.SequenceMatcher",
                        "autojunk": False,
                    },
                    metadata={
                        "candidate_a_sequence_id": str(candidate_a_pages.sequence.id),
                        "candidate_b_sequence_id": str(candidate_b_pages.sequence.id),
                    },
                )
            ],
            invocation_traces=[
                InvocationTrace(
                    node_run_id=request.node_run.id,
                    invocation_type=OCR_COMPARE_PAGES_OPERATOR_ID,
                    input_artifact_refs=[
                        artifact.ref()
                        for artifact in candidate_a_pages.artifacts
                        + candidate_b_pages.artifacts
                    ],
                    output_artifact_refs=[
                        artifact.ref()
                        for artifact in [*comparison_artifacts, metrics_artifact]
                    ],
                    provider="local",
                    model="difflib.SequenceMatcher",
                    runtime={
                        "page_count": page_count,
                        "mean_similarity_ratio": mean_similarity_ratio,
                    },
                    metadata={"output_sequence_id": str(sequence.id)},
                )
            ],
        )


class OcrSelectPagesHandler:
    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = request.node_run.metadata.get("workflow_node_config", {})
        if not isinstance(raw_config, dict):
            raise NodeRunExecutionError(
                "ocr.select_pages expected workflow_node_config metadata to be an object",
                retryable=False,
            )

        raw_selected_candidate = raw_config.get("selected_candidate", "candidate_a")
        if (
            not isinstance(raw_selected_candidate, str)
            or raw_selected_candidate not in {"candidate_a", "candidate_b"}
        ):
            raise NodeRunExecutionError(
                "ocr.select_pages requires workflow_node_config.selected_candidate "
                "to be candidate_a or candidate_b",
                retryable=False,
            )
        selected_candidate = raw_selected_candidate

        raw_decision_note = raw_config.get("decision_note")
        if raw_decision_note is not None and not isinstance(raw_decision_note, str):
            raise NodeRunExecutionError(
                "ocr.select_pages requires workflow_node_config.decision_note "
                "to be a string",
                retryable=False,
            )
        if raw_decision_note == "":
            raise NodeRunExecutionError(
                "ocr.select_pages requires workflow_node_config.decision_note "
                "to be non-empty",
                retryable=False,
            )
        decision_note = raw_decision_note

        candidate_a_pages = request.input_artifacts.get("candidate_a_pages")
        if not isinstance(candidate_a_pages, ArtifactSequenceInput):
            raise NodeRunExecutionError(
                "ocr.select_pages requires candidate_a_pages to be an artifact "
                "sequence input",
                retryable=False,
            )
        candidate_b_pages = request.input_artifacts.get("candidate_b_pages")
        if not isinstance(candidate_b_pages, ArtifactSequenceInput):
            raise NodeRunExecutionError(
                "ocr.select_pages requires candidate_b_pages to be an artifact "
                "sequence input",
                retryable=False,
            )
        if len(candidate_a_pages.artifacts) != len(candidate_b_pages.artifacts):
            raise NodeRunExecutionError(
                "ocr.select_pages requires sequences with matching page counts: "
                f"candidate_a_pages has {len(candidate_a_pages.artifacts)}, "
                f"candidate_b_pages has {len(candidate_b_pages.artifacts)}",
                retryable=False,
            )

        raw_comparison_pages = request.input_artifacts.get("comparison_pages")
        comparison_pages: ArtifactSequenceInput | None = None
        if raw_comparison_pages is not None:
            if not isinstance(raw_comparison_pages, ArtifactSequenceInput):
                raise NodeRunExecutionError(
                    "ocr.select_pages requires comparison_pages to be an artifact "
                    "sequence input when provided",
                    retryable=False,
                )
            if len(raw_comparison_pages.artifacts) != len(candidate_a_pages.artifacts):
                raise NodeRunExecutionError(
                    "ocr.select_pages requires comparison_pages to match candidate "
                    "page count: comparison_pages has "
                    f"{len(raw_comparison_pages.artifacts)}, candidates have "
                    f"{len(candidate_a_pages.artifacts)}",
                    retryable=False,
                )
            comparison_pages = raw_comparison_pages

        if selected_candidate == "candidate_a":
            selected_pages = candidate_a_pages
            rejected_pages = candidate_b_pages
        else:
            selected_pages = candidate_b_pages
            rejected_pages = candidate_a_pages

        sequence_metadata = {
            "selected_candidate": selected_candidate,
            "selected_sequence_id": str(selected_pages.sequence.id),
            "rejected_sequence_id": str(rejected_pages.sequence.id),
            "page_count": len(selected_pages.artifacts),
        }
        if comparison_pages is not None:
            sequence_metadata["comparison_sequence_id"] = str(comparison_pages.sequence.id)
        if decision_note is not None:
            sequence_metadata["decision_note"] = decision_note

        sequence = ArtifactSequence(
            artifact_type="ocr.page_result",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in selected_pages.artifacts],
            index_key=selected_pages.sequence.index_key,
            metadata=sequence_metadata,
        )
        selected_inputs = {
            "candidate_a_pages": [
                artifact.ref() for artifact in candidate_a_pages.artifacts
            ],
            "candidate_b_pages": [
                artifact.ref() for artifact in candidate_b_pages.artifacts
            ],
        }
        input_artifact_refs = [
            artifact.ref()
            for artifact in candidate_a_pages.artifacts + candidate_b_pages.artifacts
        ]
        if comparison_pages is not None:
            comparison_refs = [artifact.ref() for artifact in comparison_pages.artifacts]
            selected_inputs["comparison_pages"] = comparison_refs
            input_artifact_refs.extend(comparison_refs)

        return NodeExecutionResult(
            output_artifact_refs={"selected_pages": sequence.ref()},
            artifact_sequences=[sequence],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs=selected_inputs,
                    policies={"selected_candidate": selected_candidate},
                    metadata=sequence_metadata,
                )
            ],
            invocation_traces=[
                InvocationTrace(
                    node_run_id=request.node_run.id,
                    invocation_type=OCR_SELECT_PAGES_OPERATOR_ID,
                    input_artifact_refs=input_artifact_refs,
                    output_artifact_refs=[
                        artifact.ref() for artifact in selected_pages.artifacts
                    ],
                    provider="local",
                    model="configured_selection",
                    runtime={
                        "selected_candidate": selected_candidate,
                        "page_count": len(selected_pages.artifacts),
                    },
                    metadata={"output_sequence_id": str(sequence.id)},
                )
            ],
        )


class PromptTemplateDefineHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = _workflow_node_config(request, PROMPT_TEMPLATE_DEFINE_OPERATOR_ID)
        payload = PromptTemplatePayload(
            name=_required_string_config(
                raw_config,
                PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                "name",
            ),
            template=_required_string_config(
                raw_config,
                PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                "template",
            ),
            template_format=_enum_string_config(
                raw_config,
                PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                "template_format",
                allowed={"jinja2", "plain_text", "markdown"},
                default="jinja2",
            ),
            variables=_string_list_config(
                raw_config,
                PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                "variables",
            ),
            description=_optional_string_config(
                raw_config,
                PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                "description",
            ),
        )
        artifact = _save_spec_artifact(
            self.payload_storage,
            request,
            artifact_type="prompt.template",
            bucket="prompt-templates",
            filename="prompt-template.json",
            payload=payload,
            metadata={
                "name": payload.name,
                "template_format": payload.template_format,
                "variable_count": len(payload.variables),
            },
        )
        return _spec_node_result(
            request,
            output_name="template",
            artifact=artifact,
            policies={"template_format": payload.template_format},
            runtime={"template_length": len(payload.template)},
        )


class ExtractionSchemaDefineHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = _workflow_node_config(request, EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID)
        raw_schema = raw_config.get("json_schema")
        if not isinstance(raw_schema, dict) or not raw_schema:
            raise NodeRunExecutionError(
                "extraction.schema.define requires workflow_node_config.json_schema "
                "to be a non-empty object",
                retryable=False,
            )
        payload = ExtractionSchemaPayload(
            name=_required_string_config(
                raw_config,
                EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
                "name",
            ),
            json_schema=dict(raw_schema),
            schema_format=_enum_string_config(
                raw_config,
                EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
                "schema_format",
                allowed={"json_schema"},
                default="json_schema",
            ),
            description=_optional_string_config(
                raw_config,
                EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
                "description",
            ),
        )
        artifact = _save_spec_artifact(
            self.payload_storage,
            request,
            artifact_type="extraction.schema",
            bucket="extraction-schemas",
            filename="schema.json",
            payload=payload,
            metadata={
                "name": payload.name,
                "schema_format": payload.schema_format,
            },
        )
        return _spec_node_result(
            request,
            output_name="schema",
            artifact=artifact,
            policies={"schema_format": payload.schema_format},
            runtime={"top_level_keys": sorted(payload.json_schema)},
        )


class ModelBindingDefineHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = _workflow_node_config(request, MODEL_BINDING_DEFINE_OPERATOR_ID)
        raw_parameters = raw_config.get("parameters", {})
        if not isinstance(raw_parameters, dict):
            raise NodeRunExecutionError(
                "model.binding.define requires workflow_node_config.parameters "
                "to be an object",
                retryable=False,
            )
        _reject_sensitive_config_keys(MODEL_BINDING_DEFINE_OPERATOR_ID, raw_config)
        payload = ModelBindingPayload(
            provider=_required_string_config(
                raw_config,
                MODEL_BINDING_DEFINE_OPERATOR_ID,
                "provider",
            ),
            model=_required_string_config(
                raw_config,
                MODEL_BINDING_DEFINE_OPERATOR_ID,
                "model",
            ),
            parameters=dict(raw_parameters),
            capabilities=_string_list_config(
                raw_config,
                MODEL_BINDING_DEFINE_OPERATOR_ID,
                "capabilities",
            ),
            credential_ref=_optional_string_config(
                raw_config,
                MODEL_BINDING_DEFINE_OPERATOR_ID,
                "credential_ref",
            ),
            endpoint_ref=_optional_string_config(
                raw_config,
                MODEL_BINDING_DEFINE_OPERATOR_ID,
                "endpoint_ref",
            ),
        )
        artifact = _save_spec_artifact(
            self.payload_storage,
            request,
            artifact_type="model.binding",
            bucket="model-bindings",
            filename="model-binding.json",
            payload=payload,
            metadata={
                "provider": payload.provider,
                "model": payload.model,
                "capability_count": len(payload.capabilities),
                "has_credential_ref": payload.credential_ref is not None,
            },
        )
        return _spec_node_result(
            request,
            output_name="binding",
            artifact=artifact,
            policies={"provider": payload.provider},
            runtime={
                "model": payload.model,
                "capabilities": payload.capabilities,
                "parameter_keys": sorted(payload.parameters),
            },
        )


class InputPolicyDefineHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = _workflow_node_config(request, INPUT_POLICY_DEFINE_OPERATOR_ID)
        raw_settings = raw_config.get("settings", {})
        if not isinstance(raw_settings, dict):
            raise NodeRunExecutionError(
                "input.policy.define requires workflow_node_config.settings "
                "to be an object",
                retryable=False,
            )
        payload = InputPolicyPayload(
            name=_required_string_config(
                raw_config,
                INPUT_POLICY_DEFINE_OPERATOR_ID,
                "name",
            ),
            policy_type=_enum_string_config(
                raw_config,
                INPUT_POLICY_DEFINE_OPERATOR_ID,
                "policy_type",
                allowed={"stateless", "accumulating", "sliding_window", "custom"},
                default="stateless",
            ),
            settings=dict(raw_settings),
            applies_to=_string_list_config(
                raw_config,
                INPUT_POLICY_DEFINE_OPERATOR_ID,
                "applies_to",
            ),
            description=_optional_string_config(
                raw_config,
                INPUT_POLICY_DEFINE_OPERATOR_ID,
                "description",
            ),
        )
        artifact = _save_spec_artifact(
            self.payload_storage,
            request,
            artifact_type="input.policy",
            bucket="input-policies",
            filename="input-policy.json",
            payload=payload,
            metadata={
                "name": payload.name,
                "policy_type": payload.policy_type,
                "applies_to_count": len(payload.applies_to),
            },
        )
        return _spec_node_result(
            request,
            output_name="policy",
            artifact=artifact,
            policies={"policy_type": payload.policy_type},
            runtime={
                "setting_keys": sorted(payload.settings),
                "applies_to": payload.applies_to,
            },
        )


class StaticContextDefineHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = _workflow_node_config(request, CONTEXT_STATIC_DEFINE_OPERATOR_ID)
        raw_context = raw_config.get("context")
        if not isinstance(raw_context, dict):
            raise NodeRunExecutionError(
                "context.static.define requires workflow_node_config.context "
                "to be an object",
                retryable=False,
            )
        _reject_sensitive_config_keys(CONTEXT_STATIC_DEFINE_OPERATOR_ID, raw_config)
        payload = ContextBundlePayload(
            name=_required_string_config(
                raw_config,
                CONTEXT_STATIC_DEFINE_OPERATOR_ID,
                "name",
            ),
            context=dict(raw_context),
            applies_to=_string_list_config(
                raw_config,
                CONTEXT_STATIC_DEFINE_OPERATOR_ID,
                "applies_to",
            ),
            description=_optional_string_config(
                raw_config,
                CONTEXT_STATIC_DEFINE_OPERATOR_ID,
                "description",
            ),
        )
        artifact = _save_spec_artifact(
            self.payload_storage,
            request,
            artifact_type="context.bundle",
            bucket="context-bundles",
            filename="context-bundle.json",
            payload=payload,
            metadata={
                "name": payload.name,
                "context_keys": sorted(payload.context),
                "applies_to_count": len(payload.applies_to),
            },
        )
        return _spec_node_result(
            request,
            output_name="context",
            artifact=artifact,
            policies={"applies_to": payload.applies_to},
            runtime={"context_keys": sorted(payload.context)},
        )


class ContextualStructuredExtractionHandler:
    def __init__(
        self,
        payload_storage: ArtifactPayloadStoragePort,
        engines: Mapping[str, StructuredExtractionEnginePort] | None = None,
    ) -> None:
        self.payload_storage = payload_storage
        default_engine = LocalEchoStructuredExtractionEngine()
        provider_engine = OpenAICompatibleStructuredExtractionEngine()
        self.engines = dict(
            engines
            or {
                default_engine.engine_id: default_engine,
                "local": default_engine,
                provider_engine.engine_id: provider_engine,
                "openai": provider_engine,
            }
        )

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = _workflow_node_config(
            request,
            CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
        )
        raw_result_key = raw_config.get("result_key", "record")
        if not isinstance(raw_result_key, str) or raw_result_key == "":
            raise NodeRunExecutionError(
                "extraction.contextual_structured requires "
                "workflow_node_config.result_key to be a non-empty string",
                retryable=False,
            )
        result_key = raw_result_key

        text_pages = _required_sequence_input(request, "text")
        page_images = _optional_sequence_input(request, "pages")
        if page_images is not None and len(page_images.artifacts) != len(
            text_pages.artifacts
        ):
            raise NodeRunExecutionError(
                "extraction.contextual_structured requires pages to match text "
                f"count: pages has {len(page_images.artifacts)}, text has "
                f"{len(text_pages.artifacts)}",
                retryable=False,
            )

        schema_artifact = _required_artifact_input(request, "schema")
        template_artifact = _required_artifact_input(request, "template")
        binding_artifact = _required_artifact_input(request, "binding")
        policy_artifact = _required_artifact_input(request, "policy")
        context_artifact = _optional_artifact_input(request, "context")
        schema_payload = _load_json_model(
            self.payload_storage,
            schema_artifact,
            "schema",
            ExtractionSchemaPayload,
        )
        template_payload = _load_json_model(
            self.payload_storage,
            template_artifact,
            "template",
            PromptTemplatePayload,
        )
        binding_payload = _load_json_model(
            self.payload_storage,
            binding_artifact,
            "binding",
            ModelBindingPayload,
        )
        policy_payload = _load_json_model(
            self.payload_storage,
            policy_artifact,
            "policy",
            InputPolicyPayload,
        )
        context_payload = None
        if context_artifact is not None:
            context_payload = _load_json_model(
                self.payload_storage,
                context_artifact,
                "context",
                ContextBundlePayload,
            )
        engine = _structured_extraction_engine(self.engines, binding_payload)
        schema_validator = _extraction_schema_validator(
            schema_payload,
            schema_artifact,
        )

        model_input_artifacts: list[Artifact] = []
        model_response_artifacts: list[Artifact] = []
        page_result_artifacts: list[Artifact] = []
        invocation_traces: list[InvocationTrace] = []
        previous_records: list[dict[str, object]] = []
        for sequence_index, ocr_artifact in enumerate(text_pages.artifacts, start=1):
            ocr_payload = _load_ocr_page_result_payload(
                self.payload_storage,
                ocr_artifact,
                port_name="text",
            )
            page_artifact = (
                page_images.artifacts[sequence_index - 1]
                if page_images is not None
                else None
            )
            context = _extraction_context(
                sequence_index=sequence_index,
                ocr_payload=ocr_payload,
                ocr_artifact=ocr_artifact,
                page_artifact=page_artifact,
                schema_payload=schema_payload,
                binding_payload=binding_payload,
                policy_payload=policy_payload,
                context_payload=context_payload,
                previous_records=previous_records,
            )
            rendered_prompt = _render_prompt_template(
                template_payload,
                context,
                page_number=ocr_payload.page_number,
            )
            input_artifact_ids = _extraction_input_artifact_ids(
                ocr_artifact=ocr_artifact,
                page_artifact=page_artifact,
                schema_artifact=schema_artifact,
                template_artifact=template_artifact,
                binding_artifact=binding_artifact,
                policy_artifact=policy_artifact,
                context_artifact=context_artifact,
            )
            model_input_payload = ModelInputPayload(
                sequence_index=sequence_index,
                page_number=ocr_payload.page_number,
                rendered_prompt=rendered_prompt,
                context=context,
                prompt_template_artifact_id=str(template_artifact.id),
                extraction_schema_artifact_id=str(schema_artifact.id),
                model_binding_artifact_id=str(binding_artifact.id),
                input_policy_artifact_id=str(policy_artifact.id),
                ocr_artifact_id=str(ocr_artifact.id),
                context_bundle_artifact_id=(
                    str(context_artifact.id) if context_artifact is not None else None
                ),
                page_artifact_id=str(page_artifact.id) if page_artifact else None,
            )
            model_input_artifact = _save_json_artifact(
                self.payload_storage,
                request,
                artifact_type="model.input",
                bucket="model-inputs",
                filename=f"model-inputs/{sequence_index:04d}.json",
                payload=model_input_payload,
                metadata={
                    "sequence_index": sequence_index,
                    "page_number": ocr_payload.page_number,
                    "prompt_template_artifact_id": str(template_artifact.id),
                    "content_type": "application/json",
                },
                input_artifact_ids=input_artifact_ids,
            )

            try:
                record = engine.extract_page(
                    ContextualExtractionPageInput(
                        sequence_index=sequence_index,
                        page_number=ocr_payload.page_number,
                        page_text=ocr_payload.text,
                        rendered_prompt=rendered_prompt,
                        context=context,
                        schema=schema_payload.json_schema,
                        result_key=result_key,
                    ),
                    binding_payload,
                )
            except StructuredExtractionEngineError as exc:
                raise NodeRunExecutionError(
                    "Structured extraction failed for provider "
                    f"{binding_payload.provider!r}, model {binding_payload.model!r}, "
                    f"page {ocr_payload.page_number}: {exc}",
                    retryable=exc.retryable,
                ) from exc
            validation_errors = _schema_validation_errors(schema_validator, record)
            model_response_payload = ModelResponsePayload(
                sequence_index=sequence_index,
                page_number=ocr_payload.page_number,
                provider=binding_payload.provider,
                model=binding_payload.model,
                engine=engine.engine_id,
                response=record,
                validation_errors=validation_errors,
                model_input_artifact_id=str(model_input_artifact.id),
            )
            model_response_artifact = _save_json_artifact(
                self.payload_storage,
                request,
                artifact_type="model.response",
                bucket="model-responses",
                filename=f"model-responses/{sequence_index:04d}.json",
                payload=model_response_payload,
                metadata={
                    "sequence_index": sequence_index,
                    "page_number": ocr_payload.page_number,
                    "provider": binding_payload.provider,
                    "model": binding_payload.model,
                    "validation_error_count": len(validation_errors),
                    "content_type": "application/json",
                },
                input_artifact_ids=[model_input_artifact.id],
            )
            record_payload = ExtractionRecordResultPayload(
                sequence_index=sequence_index,
                page_number=ocr_payload.page_number,
                record=record,
                validation_errors=validation_errors,
                evidence=[
                    {
                        "artifact_id": str(ocr_artifact.id),
                        "artifact_type": ocr_artifact.artifact_type,
                        "page_number": ocr_payload.page_number,
                        "span": [0, len(ocr_payload.text)],
                    }
                ],
                model_input_artifact_id=str(model_input_artifact.id),
                model_response_artifact_id=str(model_response_artifact.id),
                ocr_artifact_id=str(ocr_artifact.id),
                page_artifact_id=str(page_artifact.id) if page_artifact else None,
            )
            page_result_artifact = _save_json_artifact(
                self.payload_storage,
                request,
                artifact_type="extraction.record_result",
                bucket="extraction-record-results",
                filename=f"record-results/{sequence_index:04d}.json",
                payload=record_payload,
                metadata={
                    "sequence_index": sequence_index,
                    "page_number": ocr_payload.page_number,
                    "validation_error_count": len(validation_errors),
                    "content_type": "application/json",
                },
                input_artifact_ids=[
                    ocr_artifact.id,
                    model_input_artifact.id,
                    model_response_artifact.id,
                ],
            )

            model_input_artifacts.append(model_input_artifact)
            model_response_artifacts.append(model_response_artifact)
            page_result_artifacts.append(page_result_artifact)
            previous_records.append(record)
            invocation_input_refs = [
                ocr_artifact.ref(),
                schema_artifact.ref(),
                template_artifact.ref(),
                binding_artifact.ref(),
                policy_artifact.ref(),
            ]
            if context_artifact is not None:
                invocation_input_refs.append(context_artifact.ref())
            if page_artifact is not None:
                invocation_input_refs.append(page_artifact.ref())
            invocation_traces.append(
                InvocationTrace(
                    node_run_id=request.node_run.id,
                    invocation_type=CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
                    input_artifact_refs=invocation_input_refs,
                    output_artifact_refs=[
                        model_response_artifact.ref(),
                        page_result_artifact.ref(),
                    ],
                    provider=binding_payload.provider,
                    model=binding_payload.model,
                    request_ref=model_input_artifact.payload_ref,
                    response_ref=model_response_artifact.payload_ref,
                    runtime={
                        "sequence_index": sequence_index,
                        "page_number": ocr_payload.page_number,
                        "engine": engine.engine_id,
                        "validation_error_count": len(validation_errors),
                    },
                    metadata={
                        "model_input_artifact_id": str(model_input_artifact.id),
                        "record_result_artifact_id": str(page_result_artifact.id),
                    },
                )
            )

        page_result_sequence = ArtifactSequence(
            artifact_type="extraction.record_result",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in page_result_artifacts],
            index_key=text_pages.sequence.index_key,
            metadata={"page_count": len(page_result_artifacts)},
        )
        model_input_sequence = ArtifactSequence(
            artifact_type="model.input",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in model_input_artifacts],
            index_key=text_pages.sequence.index_key,
            metadata={"page_count": len(model_input_artifacts)},
        )
        model_response_sequence = ArtifactSequence(
            artifact_type="model.response",
            schema_version=1,
            item_refs=[artifact.ref() for artifact in model_response_artifacts],
            index_key=text_pages.sequence.index_key,
            metadata={"page_count": len(model_response_artifacts)},
        )
        validation_error_count = sum(
            int(artifact.metadata["validation_error_count"])
            for artifact in page_result_artifacts
        )
        document_payload = ExtractionDocumentResultPayload(
            page_count=len(page_result_artifacts),
            record_count=len(previous_records),
            records=previous_records,
            validation_error_count=validation_error_count,
            page_result_artifact_ids=[
                str(artifact.id) for artifact in page_result_artifacts
            ],
            model_input_sequence_id=str(model_input_sequence.id),
            model_response_sequence_id=str(model_response_sequence.id),
            provider=binding_payload.provider,
            model=binding_payload.model,
            policy_type=policy_payload.policy_type,
        )
        document_artifact = _save_json_artifact(
            self.payload_storage,
            request,
            artifact_type="extraction.document_result",
            bucket="extraction-document-results",
            filename="document-result.json",
            payload=document_payload,
            metadata={
                "page_count": document_payload.page_count,
                "record_count": document_payload.record_count,
                "validation_error_count": validation_error_count,
                "provider": binding_payload.provider,
                "model": binding_payload.model,
                "content_type": "application/json",
            },
            input_artifact_ids=[artifact.id for artifact in page_result_artifacts],
        )

        selected_inputs = {
            "text": [artifact.ref() for artifact in text_pages.artifacts],
            "schema": schema_artifact.ref(),
            "template": template_artifact.ref(),
            "binding": binding_artifact.ref(),
            "policy": policy_artifact.ref(),
        }
        if context_artifact is not None:
            selected_inputs["context"] = context_artifact.ref()
        if page_images is not None:
            selected_inputs["pages"] = [
                artifact.ref() for artifact in page_images.artifacts
            ]

        return NodeExecutionResult(
            output_artifact_refs={
                "page_results": page_result_sequence.ref(),
                "document_result": document_artifact.ref(),
                "model_inputs": model_input_sequence.ref(),
                "model_responses": model_response_sequence.ref(),
            },
            artifacts=[
                *model_input_artifacts,
                *model_response_artifacts,
                *page_result_artifacts,
                document_artifact,
            ],
            artifact_sequences=[
                page_result_sequence,
                model_input_sequence,
                model_response_sequence,
            ],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs=selected_inputs,
                    policies={
                        "policy_type": policy_payload.policy_type,
                        "settings": policy_payload.settings,
                    },
                    metadata={
                        "text_sequence_id": str(text_pages.sequence.id),
                        "page_count": len(text_pages.artifacts),
                    },
                )
            ],
            invocation_traces=invocation_traces,
        )


class ExportDatasetHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        raw_config = _workflow_node_config(request, EXPORT_DATASET_OPERATOR_ID)
        export_format = _enum_string_config(
            raw_config,
            EXPORT_DATASET_OPERATOR_ID,
            "format",
            allowed={"json", "jsonl", "csv"},
            default="json",
        )
        document_artifact = _required_artifact_input(
            request,
            "document",
            operator_id=EXPORT_DATASET_OPERATOR_ID,
        )
        document_payload = _load_json_model(
            self.payload_storage,
            document_artifact,
            "document",
            ExtractionDocumentResultPayload,
        )
        filename = _export_filename(raw_config, export_format)
        payload = _export_dataset_bytes(
            export_format,
            document_payload,
            document_artifact,
        )
        artifact = _save_bytes_artifact(
            self.payload_storage,
            request,
            artifact_type="export.dataset",
            bucket="export-datasets",
            filename=filename,
            payload=payload,
            content_type=_export_content_type(export_format),
            metadata={
                "format": export_format,
                "filename": filename,
                "record_count": document_payload.record_count,
                "source_artifact_id": str(document_artifact.id),
            },
            input_artifact_ids=[document_artifact.id],
        )
        return NodeExecutionResult(
            output_artifact_refs={"dataset": artifact.ref()},
            artifacts=[artifact],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs={"document": document_artifact.ref()},
                    policies={"format": export_format},
                    metadata={"record_count": document_payload.record_count},
                )
            ],
            invocation_traces=[
                InvocationTrace(
                    node_run_id=request.node_run.id,
                    invocation_type=EXPORT_DATASET_OPERATOR_ID,
                    input_artifact_refs=[document_artifact.ref()],
                    output_artifact_refs=[artifact.ref()],
                    provider="local",
                    model="dataset-exporter",
                    runtime={
                        "format": export_format,
                        "record_count": document_payload.record_count,
                    },
                    metadata={"filename": filename},
                )
            ],
        )


class SchemaValidationHandler:
    def __init__(self, payload_storage: ArtifactPayloadStoragePort) -> None:
        self.payload_storage = payload_storage

    async def execute(self, request: NodeExecutionRequest) -> NodeExecutionResult:
        _workflow_node_config(request, SCHEMA_VALIDATION_OPERATOR_ID)
        document_artifact = _required_artifact_input(
            request,
            "document",
            operator_id=SCHEMA_VALIDATION_OPERATOR_ID,
        )
        schema_artifact = _required_artifact_input(
            request,
            "schema",
            operator_id=SCHEMA_VALIDATION_OPERATOR_ID,
        )
        document_payload = _load_json_model(
            self.payload_storage,
            document_artifact,
            "document",
            ExtractionDocumentResultPayload,
        )
        schema_payload = _load_json_model(
            self.payload_storage,
            schema_artifact,
            "schema",
            ExtractionSchemaPayload,
        )
        schema_validator = _extraction_schema_validator(
            schema_payload,
            schema_artifact,
        )

        errors: list[dict[str, object]] = []
        invalid_record_indexes: set[int] = set()
        for record_index, record in enumerate(document_payload.records, start=1):
            record_errors = _record_validation_errors(
                schema_validator,
                record,
                record_index,
            )
            if record_errors:
                invalid_record_indexes.add(record_index)
            errors.extend(record_errors)

        record_count = len(document_payload.records)
        invalid_record_count = len(invalid_record_indexes)
        valid_record_count = record_count - invalid_record_count
        validation_payload = ValidationResultPayload(
            source_artifact_id=str(document_artifact.id),
            schema_artifact_id=str(schema_artifact.id),
            record_count=record_count,
            valid_record_count=valid_record_count,
            invalid_record_count=invalid_record_count,
            valid=errors == [],
            error_count=len(errors),
            errors=errors,
        )
        validation_artifact = _save_json_artifact(
            self.payload_storage,
            request,
            artifact_type="validation.result",
            bucket="validation-results",
            filename="validation-result.json",
            payload=validation_payload,
            metadata={
                "source_artifact_id": str(document_artifact.id),
                "schema_artifact_id": str(schema_artifact.id),
                "record_count": record_count,
                "valid_record_count": valid_record_count,
                "invalid_record_count": invalid_record_count,
                "valid": validation_payload.valid,
                "error_count": len(errors),
            },
            input_artifact_ids=[document_artifact.id, schema_artifact.id],
        )
        metrics_payload = EvaluationMetricsPayload(
            metric_family="schema_validation",
            metrics={
                "record_count": record_count,
                "valid_record_count": valid_record_count,
                "invalid_record_count": invalid_record_count,
                "error_count": len(errors),
                "valid": validation_payload.valid,
            },
            source_artifact_ids=[str(document_artifact.id), str(schema_artifact.id)],
            metadata={"validator": "jsonschema.Draft202012Validator"},
        )
        metrics_artifact = _save_json_artifact(
            self.payload_storage,
            request,
            artifact_type="evaluation.metrics",
            bucket="evaluation-metrics",
            filename="schema-validation-metrics.json",
            payload=metrics_payload,
            metadata={
                "metric_family": "schema_validation",
                "record_count": record_count,
                "valid_record_count": valid_record_count,
                "invalid_record_count": invalid_record_count,
                "error_count": len(errors),
                "valid": validation_payload.valid,
            },
            input_artifact_ids=[document_artifact.id, schema_artifact.id],
        )
        return NodeExecutionResult(
            output_artifact_refs={
                "validation": validation_artifact.ref(),
                "metrics": metrics_artifact.ref(),
            },
            artifacts=[validation_artifact, metrics_artifact],
            input_assembly_traces=[
                InputAssemblyTrace(
                    node_run_id=request.node_run.id,
                    selected_inputs={
                        "document": document_artifact.ref(),
                        "schema": schema_artifact.ref(),
                    },
                    policies={"validator": "jsonschema.Draft202012Validator"},
                    metadata={
                        "record_count": record_count,
                        "error_count": len(errors),
                    },
                )
            ],
            invocation_traces=[
                InvocationTrace(
                    node_run_id=request.node_run.id,
                    invocation_type=SCHEMA_VALIDATION_OPERATOR_ID,
                    input_artifact_refs=[
                        document_artifact.ref(),
                        schema_artifact.ref(),
                    ],
                    output_artifact_refs=[
                        validation_artifact.ref(),
                        metrics_artifact.ref(),
                    ],
                    provider="local",
                    model="jsonschema",
                    runtime={
                        "record_count": record_count,
                        "error_count": len(errors),
                        "valid": validation_payload.valid,
                    },
                )
            ],
        )


def _workflow_node_config(
    request: NodeExecutionRequest,
    operator_id: str,
) -> dict[str, object]:
    raw_config = request.node_run.metadata.get("workflow_node_config", {})
    if not isinstance(raw_config, dict):
        raise NodeRunExecutionError(
            f"{operator_id} expected workflow_node_config metadata to be an object",
            retryable=False,
        )
    return raw_config


def _required_string_config(
    raw_config: dict[str, object],
    operator_id: str,
    field_name: str,
) -> str:
    value = raw_config.get(field_name)
    if not isinstance(value, str) or value == "":
        raise NodeRunExecutionError(
            f"{operator_id} requires workflow_node_config.{field_name} "
            "to be a non-empty string",
            retryable=False,
        )
    return value


def _optional_string_config(
    raw_config: dict[str, object],
    operator_id: str,
    field_name: str,
) -> str | None:
    value = raw_config.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or value == "":
        raise NodeRunExecutionError(
            f"{operator_id} requires workflow_node_config.{field_name} "
            "to be a non-empty string when provided",
            retryable=False,
        )
    return value


def _enum_string_config(
    raw_config: dict[str, object],
    operator_id: str,
    field_name: str,
    *,
    allowed: set[str],
    default: str,
) -> str:
    value = raw_config.get(field_name, default)
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise NodeRunExecutionError(
            f"{operator_id} requires workflow_node_config.{field_name} "
            f"to be one of: {choices}",
            retryable=False,
        )
    return value


def _string_list_config(
    raw_config: dict[str, object],
    operator_id: str,
    field_name: str,
) -> list[str]:
    value = raw_config.get(field_name, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or item == "" for item in value
    ):
        raise NodeRunExecutionError(
            f"{operator_id} requires workflow_node_config.{field_name} "
            "to be a list of non-empty strings",
            retryable=False,
        )
    return list(value)


def _required_artifact_input(
    request: NodeExecutionRequest,
    port_name: str,
    *,
    operator_id: str = CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
) -> Artifact:
    artifact = request.input_artifacts.get(port_name)
    if not isinstance(artifact, Artifact):
        raise NodeRunExecutionError(
            f"{operator_id} requires {port_name} to be an artifact input",
            retryable=False,
        )
    return artifact


def _optional_artifact_input(
    request: NodeExecutionRequest,
    port_name: str,
) -> Artifact | None:
    artifact = request.input_artifacts.get(port_name)
    if artifact is None:
        return None
    if not isinstance(artifact, Artifact):
        raise NodeRunExecutionError(
            "extraction.contextual_structured requires "
            f"{port_name} to be an artifact input when provided",
            retryable=False,
        )
    return artifact


def _required_sequence_input(
    request: NodeExecutionRequest,
    port_name: str,
) -> ArtifactSequenceInput:
    sequence = request.input_artifacts.get(port_name)
    if not isinstance(sequence, ArtifactSequenceInput):
        raise NodeRunExecutionError(
            "extraction.contextual_structured requires "
            f"{port_name} to be an artifact sequence input",
            retryable=False,
        )
    return sequence


def _optional_sequence_input(
    request: NodeExecutionRequest,
    port_name: str,
) -> ArtifactSequenceInput | None:
    sequence = request.input_artifacts.get(port_name)
    if sequence is None:
        return None
    if not isinstance(sequence, ArtifactSequenceInput):
        raise NodeRunExecutionError(
            "extraction.contextual_structured requires "
            f"{port_name} to be an artifact sequence input when provided",
            retryable=False,
        )
    return sequence


def _structured_extraction_engine(
    engines: Mapping[str, StructuredExtractionEnginePort],
    binding: ModelBindingPayload,
) -> StructuredExtractionEnginePort:
    engine_keys = [
        f"{binding.provider}.{binding.model}",
        binding.model,
        binding.provider,
    ]
    for engine_key in engine_keys:
        engine = engines.get(engine_key)
        if engine is not None:
            return engine
    raise NodeRunExecutionError(
        "extraction.contextual_structured has no structured extraction engine "
        f"registered for provider={binding.provider!r} model={binding.model!r}",
        retryable=False,
    )


def _extraction_schema_validator(
    schema_payload: ExtractionSchemaPayload,
    schema_artifact: Artifact,
) -> Draft202012Validator:
    try:
        Draft202012Validator.check_schema(schema_payload.json_schema)
    except JsonSchemaError as exc:
        raise NodeRunExecutionError(
            "Invalid extraction schema artifact "
            f"{schema_artifact.id}: {exc.message}",
            retryable=False,
        ) from exc
    return Draft202012Validator(schema_payload.json_schema)


def _schema_validation_errors(
    validator: Draft202012Validator,
    record: dict[str, object],
) -> list[str]:
    return [
        error.message
        for error in sorted(
            validator.iter_errors(record),
            key=lambda item: list(item.path),
        )
    ]


def _record_validation_errors(
    validator: Draft202012Validator,
    record: dict[str, object],
    record_index: int,
) -> list[dict[str, object]]:
    return [
        {
            "record_index": record_index,
            "message": error.message,
            "path": list(error.path),
            "schema_path": list(error.schema_path),
            "validator": error.validator,
        }
        for error in sorted(
            validator.iter_errors(record),
            key=lambda item: list(item.path),
        )
    ]


def _extraction_context(
    *,
    sequence_index: int,
    ocr_payload: OcrPageResultPayload,
    ocr_artifact: Artifact,
    page_artifact: Artifact | None,
    schema_payload: ExtractionSchemaPayload,
    binding_payload: ModelBindingPayload,
    policy_payload: InputPolicyPayload,
    context_payload: ContextBundlePayload | None,
    previous_records: list[dict[str, object]],
) -> dict[str, object]:
    visible_previous_records = _visible_previous_records(
        policy_payload,
        previous_records,
    )
    return {
        "CURRENT_PAGE_TEXT": ocr_payload.text,
        "CURRENT_PAGE_NUMBER": ocr_payload.page_number,
        "SEQUENCE_INDEX": sequence_index,
        "OCR_PAGE": ocr_payload.model_dump(mode="json"),
        "OCR_ARTIFACT_ID": str(ocr_artifact.id),
        "PAGE_ARTIFACT_ID": str(page_artifact.id) if page_artifact else None,
        "PREVIOUS_RECORD": (
            visible_previous_records[-1] if visible_previous_records else None
        ),
        "PREVIOUS_RECORDS": visible_previous_records,
        "SCHEMA": schema_payload.json_schema,
        "MODEL": binding_payload.model_dump(mode="json"),
        "POLICY": policy_payload.model_dump(mode="json"),
        "STATIC_CONTEXT": (
            context_payload.context if context_payload is not None else {}
        ),
        "CONTEXT_BUNDLE": (
            context_payload.model_dump(mode="json")
            if context_payload is not None
            else None
        ),
    }


def _visible_previous_records(
    policy_payload: InputPolicyPayload,
    previous_records: list[dict[str, object]],
) -> list[dict[str, object]]:
    if policy_payload.policy_type == "stateless":
        return []
    if policy_payload.policy_type != "sliding_window":
        return list(previous_records)

    raw_window_size = policy_payload.settings.get("window_size", 5)
    if type(raw_window_size) is not int or raw_window_size <= 0:
        raise NodeRunExecutionError(
            "input.policy sliding_window requires settings.window_size "
            "to be a positive integer",
            retryable=False,
        )
    return previous_records[-raw_window_size:]


def _render_prompt_template(
    template_payload: PromptTemplatePayload,
    context: dict[str, object],
    *,
    page_number: int,
) -> str:
    if template_payload.template_format != "jinja2":
        return template_payload.template
    try:
        template = Environment(undefined=StrictUndefined).from_string(
            template_payload.template
        )
        return template.render(context)
    except TemplateError as exc:
        raise NodeRunExecutionError(
            "Failed to render prompt.template for page "
            f"{page_number}: {exc}",
            retryable=False,
        ) from exc


def _extraction_input_artifact_ids(
    *,
    ocr_artifact: Artifact,
    page_artifact: Artifact | None,
    schema_artifact: Artifact,
    template_artifact: Artifact,
    binding_artifact: Artifact,
    policy_artifact: Artifact,
    context_artifact: Artifact | None,
) -> list[UUID]:
    artifact_ids = [
        ocr_artifact.id,
        schema_artifact.id,
        template_artifact.id,
        binding_artifact.id,
        policy_artifact.id,
    ]
    if context_artifact is not None:
        artifact_ids.append(context_artifact.id)
    if page_artifact is not None:
        artifact_ids.append(page_artifact.id)
    return artifact_ids


def _export_filename(raw_config: dict[str, object], export_format: str) -> str:
    filename = _optional_string_config(
        raw_config,
        EXPORT_DATASET_OPERATOR_ID,
        "filename",
    )
    if filename is None:
        return f"dataset.{export_format}"
    if "/" in filename or "\\" in filename:
        raise NodeRunExecutionError(
            "export.dataset requires workflow_node_config.filename to be a "
            "filename, not a path",
            retryable=False,
        )
    return filename


def _export_content_type(export_format: str) -> str:
    if export_format == "json":
        return "application/json"
    if export_format == "jsonl":
        return "application/x-ndjson"
    return "text/csv"


def _export_dataset_bytes(
    export_format: str,
    document_payload: ExtractionDocumentResultPayload,
    document_artifact: Artifact,
) -> bytes:
    if export_format == "json":
        payload = ExportDatasetPayload(
            format=export_format,
            source_artifact_id=str(document_artifact.id),
            record_count=document_payload.record_count,
            records=document_payload.records,
            metadata={
                "page_count": document_payload.page_count,
                "validation_error_count": document_payload.validation_error_count,
                "provider": document_payload.provider,
                "model": document_payload.model,
                "policy_type": document_payload.policy_type,
            },
        )
        return payload.model_dump_json(indent=2).encode("utf-8")
    if export_format == "jsonl":
        lines = [
            json.dumps(record, sort_keys=True, ensure_ascii=False)
            for record in document_payload.records
        ]
        text = "\n".join(lines)
        if text:
            text += "\n"
        return text.encode("utf-8")
    return _csv_dataset_bytes(document_payload.records)


def _csv_dataset_bytes(records: list[dict[str, object]]) -> bytes:
    fieldnames = _csv_fieldnames(records)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {fieldname: _csv_cell(record.get(fieldname)) for fieldname in fieldnames}
        )
    return stream.getvalue().encode("utf-8")


def _csv_fieldnames(records: list[dict[str, object]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record:
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    return fieldnames


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if type(value) in {str, int, float, bool}:
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _reject_sensitive_config_keys(operator_id: str, value: object) -> None:
    sensitive_keys = {
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "refresh_token",
        "bearer_token",
        "client_secret",
        "private_key",
        "password",
        "secret",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in sensitive_keys:
                raise NodeRunExecutionError(
                    f"{operator_id} refuses to persist sensitive field {key!r}; "
                    "use credential_ref instead",
                    retryable=False,
                )
            _reject_sensitive_config_keys(operator_id, item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_config_keys(operator_id, item)


def _redacted_config(value: dict[str, object]) -> dict[str, object]:
    return {
        key: _redacted_value(key, item)
        for key, item in value.items()
        if isinstance(key, str)
    }


def _redacted_value(key: str, value: object) -> object:
    lowered = key.lower()
    sensitive_fragments = (
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "refresh_token",
        "bearer_token",
        "client_secret",
        "private_key",
        "password",
        "secret",
    )
    if any(fragment in lowered for fragment in sensitive_fragments):
        return "<redacted>"
    if isinstance(value, dict):
        return {
            nested_key: _redacted_value(nested_key, nested_value)
            for nested_key, nested_value in value.items()
            if isinstance(nested_key, str)
        }
    if isinstance(value, list):
        return [
            _redacted_value(key, item) if isinstance(item, dict | list) else item
            for item in value
        ]
    return value


def _save_spec_artifact(
    payload_storage: ArtifactPayloadStoragePort,
    request: NodeExecutionRequest,
    *,
    artifact_type: str,
    bucket: str,
    filename: str,
    payload: BaseModel,
    metadata: dict[str, object],
) -> Artifact:
    return _save_json_artifact(
        payload_storage,
        request,
        artifact_type=artifact_type,
        bucket=bucket,
        filename=filename,
        payload=payload,
        metadata=metadata,
        input_artifact_ids=[],
    )


def _save_json_artifact(
    payload_storage: ArtifactPayloadStoragePort,
    request: NodeExecutionRequest,
    *,
    artifact_type: str,
    bucket: str,
    filename: str,
    payload: BaseModel,
    metadata: dict[str, object],
    input_artifact_ids: list[UUID],
) -> Artifact:
    payload_bytes = payload.model_dump_json(indent=2).encode("utf-8")
    key = (
        f"workflow-runs/{request.node_run.workflow_run_id}/"
        f"node-runs/{request.node_run.id}/{filename}"
    )
    stored = payload_storage.save(
        SaveArtifactPayloadCommand(
            bucket=bucket,
            key=key,
            payload=payload_bytes,
            overwrite=True,
        )
    )
    return Artifact(
        artifact_type=artifact_type,
        schema_version=1,
        workflow_run_id=request.node_run.workflow_run_id,
        producer_node_run_id=request.node_run.id,
        payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
        producer_operator_id=request.node_run.operator_id,
        producer_operator_version=request.node_run.operator_version,
        input_artifact_ids=input_artifact_ids,
        content_hash=stored.sha256,
        metadata={
            **metadata,
            "content_type": "application/json",
            "byte_size": stored.byte_size,
        },
    )


def _save_bytes_artifact(
    payload_storage: ArtifactPayloadStoragePort,
    request: NodeExecutionRequest,
    *,
    artifact_type: str,
    bucket: str,
    filename: str,
    payload: bytes,
    content_type: str,
    metadata: dict[str, object],
    input_artifact_ids: list[UUID],
) -> Artifact:
    key = (
        f"workflow-runs/{request.node_run.workflow_run_id}/"
        f"node-runs/{request.node_run.id}/{filename}"
    )
    stored = payload_storage.save(
        SaveArtifactPayloadCommand(
            bucket=bucket,
            key=key,
            payload=payload,
            overwrite=True,
        )
    )
    return Artifact(
        artifact_type=artifact_type,
        schema_version=1,
        workflow_run_id=request.node_run.workflow_run_id,
        producer_node_run_id=request.node_run.id,
        payload_ref=artifact_payload_ref(bucket=stored.bucket, key=stored.key),
        producer_operator_id=request.node_run.operator_id,
        producer_operator_version=request.node_run.operator_version,
        input_artifact_ids=input_artifact_ids,
        content_hash=stored.sha256,
        metadata={
            **metadata,
            "content_type": content_type,
            "byte_size": stored.byte_size,
        },
    )


def _spec_node_result(
    request: NodeExecutionRequest,
    *,
    output_name: str,
    artifact: Artifact,
    policies: dict[str, object],
    runtime: dict[str, object],
) -> NodeExecutionResult:
    return NodeExecutionResult(
        output_artifact_refs={output_name: artifact.ref()},
        artifacts=[artifact],
        input_assembly_traces=[
            InputAssemblyTrace(
                node_run_id=request.node_run.id,
                selected_inputs={},
                policies=policies,
            )
        ],
        invocation_traces=[
            InvocationTrace(
                node_run_id=request.node_run.id,
                invocation_type=request.node_run.operator_id,
                output_artifact_refs=[artifact.ref()],
                provider="local",
                model="configured_artifact",
                runtime=runtime,
            )
        ],
    )


def _load_artifact_payload(
    payload_storage: ArtifactPayloadStoragePort,
    artifact: Artifact,
    port_name: str,
) -> bytes:
    try:
        location = parse_artifact_payload_ref(artifact.payload_ref)
        return payload_storage.load(location.bucket, location.key).payload
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise NodeRunExecutionError(
            "Failed to load artifact payload for port "
            f"{port_name} and artifact {artifact.id}: {exc}",
            retryable=False,
        ) from exc


def _load_json_model(
    payload_storage: ArtifactPayloadStoragePort,
    artifact: Artifact,
    port_name: str,
    model_type: type[PayloadModelT],
) -> PayloadModelT:
    payload = _load_artifact_payload(payload_storage, artifact, port_name)
    try:
        return model_type.model_validate_json(payload)
    except ValidationError as exc:
        raise NodeRunExecutionError(
            "Invalid JSON payload for port "
            f"{port_name} and artifact {artifact.id}: {exc}",
            retryable=False,
        ) from exc


def _load_ocr_page_result_payload(
    payload_storage: ArtifactPayloadStoragePort,
    artifact: Artifact,
    port_name: str,
) -> OcrPageResultPayload:
    payload = _load_artifact_payload(payload_storage, artifact, port_name)
    try:
        return OcrPageResultPayload.model_validate_json(payload)
    except ValidationError as exc:
        raise NodeRunExecutionError(
            "Invalid OCR page-result payload for port "
            f"{port_name} and artifact {artifact.id}: {exc}",
            retryable=False,
        ) from exc


def _page_number(artifact: Artifact, sequence_index: int) -> int:
    raw_page_number = artifact.metadata.get("page_number")
    if type(raw_page_number) is int and raw_page_number > 0:
        return raw_page_number
    return sequence_index


def builtin_node_handlers(
    payload_storage: ArtifactPayloadStoragePort | None = None,
) -> dict[tuple[str, str], NodeRunHandler]:
    handlers: dict[tuple[str, str], NodeRunHandler] = {
        (DEBUG_EMIT_TEXT_OPERATOR_ID, DEBUG_EMIT_TEXT_OPERATOR_VERSION): EmitTextHandler(),
        (OCR_SELECT_PAGES_OPERATOR_ID, OCR_SELECT_PAGES_OPERATOR_VERSION): (
            OcrSelectPagesHandler()
        ),
    }
    if payload_storage is not None:
        handlers[
            (
                CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_ID,
                CONTEXTUAL_STRUCTURED_EXTRACTION_OPERATOR_VERSION,
            )
        ] = ContextualStructuredExtractionHandler(payload_storage)
        handlers[
            (
                CONTEXT_STATIC_DEFINE_OPERATOR_ID,
                CONTEXT_STATIC_DEFINE_OPERATOR_VERSION,
            )
        ] = StaticContextDefineHandler(payload_storage)
        handlers[
            (
                EXTRACTION_SCHEMA_DEFINE_OPERATOR_ID,
                EXTRACTION_SCHEMA_DEFINE_OPERATOR_VERSION,
            )
        ] = ExtractionSchemaDefineHandler(payload_storage)
        handlers[(EXPORT_DATASET_OPERATOR_ID, EXPORT_DATASET_OPERATOR_VERSION)] = (
            ExportDatasetHandler(payload_storage)
        )
        handlers[
            (
                INPUT_POLICY_DEFINE_OPERATOR_ID,
                INPUT_POLICY_DEFINE_OPERATOR_VERSION,
            )
        ] = InputPolicyDefineHandler(payload_storage)
        handlers[
            (
                MODEL_BINDING_DEFINE_OPERATOR_ID,
                MODEL_BINDING_DEFINE_OPERATOR_VERSION,
            )
        ] = ModelBindingDefineHandler(payload_storage)
        handlers[(OCR_COMPARE_PAGES_OPERATOR_ID, OCR_COMPARE_PAGES_OPERATOR_VERSION)] = (
            OcrComparePagesHandler(payload_storage)
        )
        handlers[(OCR_COLLECT_PAGES_OPERATOR_ID, OCR_COLLECT_PAGES_OPERATOR_VERSION)] = (
            OcrCollectPagesHandler(payload_storage)
        )
        handlers[(OCR_EXTRACT_PAGE_OPERATOR_ID, OCR_EXTRACT_PAGE_OPERATOR_VERSION)] = (
            OcrExtractPageHandler(payload_storage)
        )
        handlers[(OCR_EXTRACT_PAGES_OPERATOR_ID, OCR_EXTRACT_PAGES_OPERATOR_VERSION)] = (
            OcrExtractPagesHandler(payload_storage)
        )
        handlers[
            (
                PROMPT_TEMPLATE_DEFINE_OPERATOR_ID,
                PROMPT_TEMPLATE_DEFINE_OPERATOR_VERSION,
            )
        ] = PromptTemplateDefineHandler(payload_storage)
        handlers[(SCHEMA_VALIDATION_OPERATOR_ID, SCHEMA_VALIDATION_OPERATOR_VERSION)] = (
            SchemaValidationHandler(payload_storage)
        )
    return handlers
