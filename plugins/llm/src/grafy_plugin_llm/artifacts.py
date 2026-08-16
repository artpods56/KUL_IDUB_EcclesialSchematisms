from typing import Literal, Self, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from grafy_core.artifacts import (
    ArtifactTypeKey,
    ArtifactTypeSpec,
    JsonObject,
)
from grafy_core.operators.schemas import (
    parse_json_schema,
    validate_json_schema_value,
)


class StructuredResponsePayload(BaseModel):
    """Portable structured completion plus provider audit information."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    value: JsonObject
    model: str = Field(min_length=1)
    provider: Literal["mistral"] = "mistral"
    json_schema: StrictStr = Field(
        alias="schema",
        description="JSON Schema text requested from the provider.",
    )
    schema_name: StrictStr = Field(
        min_length=1,
        description="Provider-facing name used for the schema request.",
    )
    schema_description: StrictStr = Field(
        default="",
        description="Provider-facing description used for the schema request.",
    )
    schema_strict: bool = Field(
        default=True,
        description="Whether the provider was asked to enforce the schema strictly.",
    )
    message_count: int = Field(ge=1)
    source_image_artifact_ids: list[UUID] = Field(default_factory=list)
    usage: JsonObject = Field(default_factory=dict)
    raw_response: JsonObject

    @model_validator(mode="after")
    def validate_json_schema(self) -> Self:
        parse_json_schema(
            self.json_schema,
            context=f"structured response schema {self.schema_name!r}",
        )
        return self


STRUCTURED_RESPONSE = ArtifactTypeSpec(
    key=ArtifactTypeKey("llm.structured_response", 1),
    title="Structured LLM response",
    payload_schema=cast(
        JsonObject,
        StructuredResponsePayload.model_json_schema(),
    ),
)


class CompletionPayload(BaseModel):
    """Provider-neutral Chat Completions result safe for artifact persistence."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    content: StrictStr = Field(
        description="Assistant message content returned by the provider.",
    )
    structured_value: JsonObject | None = Field(
        default=None,
        description="Parsed and validated object when a JSON Schema was requested.",
    )
    model: StrictStr = Field(min_length=1)
    protocol: Literal["openai_chat_completions"] = "openai_chat_completions"
    base_url: StrictStr = Field(
        min_length=1,
        description="OpenAI-compatible API base URL used for the request.",
    )
    response_id: StrictStr | None = None
    finish_reason: StrictStr | None = None
    message_count: int = Field(ge=1)
    source_image_artifact_ids: list[UUID] = Field(default_factory=list)
    json_schema: StrictStr | None = Field(
        default=None,
        alias="schema",
        description="JSON Schema text requested from the provider, when present.",
    )
    schema_name: StrictStr | None = Field(default=None, min_length=1)
    schema_strict: bool | None = None
    usage: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structured_result(self) -> Self:
        if self.json_schema is None:
            if self.structured_value is not None:
                raise ValueError("structured_value requires a JSON Schema")
            if self.schema_name is not None or self.schema_strict is not None:
                raise ValueError("Schema metadata requires a JSON Schema")
            return self

        if self.structured_value is None:
            raise ValueError("A JSON Schema completion requires structured_value")
        if self.schema_name is None or self.schema_strict is None:
            raise ValueError("A JSON Schema completion requires schema metadata")
        parse_json_schema(
            self.json_schema,
            context=f"completion schema {self.schema_name!r}",
        )
        validate_json_schema_value(self.json_schema, self.structured_value)
        return self


COMPLETION = ArtifactTypeSpec(
    key=ArtifactTypeKey("llm.completion", 1),
    title="LLM completion",
    payload_schema=cast(
        JsonObject,
        CompletionPayload.model_json_schema(),
    ),
)
