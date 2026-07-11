from sqlalchemy import JSON, Boolean, DateTime, Integer, MetaData, String, Table
from sqlalchemy import Column

metadata = MetaData()

projects = Table(
    "projects",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("description", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

sources = Table(
    "sources",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("name", String, nullable=False),
    Column("description", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

source_items = Table(
    "source_items",
    metadata,
    Column("id", String, primary_key=True),
    Column("source_id", String, nullable=False, index=True),
    Column("order", Integer, nullable=False),
    Column("text", String, nullable=True),
    Column("image_path", String, nullable=True),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

output_schemas = Table(
    "output_schemas",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("name", String, nullable=False),
    Column("description", String, nullable=True),
    Column("json_schema", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

recipes = Table(
    "recipes",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("schema_id", String, nullable=False),
    Column("name", String, nullable=False),
    Column("description", String, nullable=True),
    Column("config", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", String, primary_key=True),
    Column("project_id", String, nullable=False, index=True),
    Column("source_id", String, nullable=False),
    Column("recipe_id", String, nullable=False),
    Column("status", String, nullable=False),
    Column("error", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

job_items = Table(
    "job_items",
    metadata,
    Column("id", String, primary_key=True),
    Column("job_id", String, nullable=False, index=True),
    Column("source_item_id", String, nullable=False),
    Column("order", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("structured_output", JSON, nullable=True),
    Column("context_trace", JSON, nullable=True),
    Column("error", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

workflow_definitions = Table(
    "workflow_definitions",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("description", String, nullable=True),
    Column("nodes", JSON, nullable=False),
    Column("edges", JSON, nullable=False),
    Column("declared_inputs", JSON, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

workflow_versions = Table(
    "workflow_versions",
    metadata,
    Column("id", String, primary_key=True),
    Column("workflow_definition_id", String, nullable=False, index=True),
    Column("version_number", Integer, nullable=False),
    Column("definition_snapshot", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("created_by", String, nullable=True),
    Column("change_note", String, nullable=True),
)

workflow_runs = Table(
    "workflow_runs",
    metadata,
    Column("id", String, primary_key=True),
    Column("workflow_version_id", String, nullable=False, index=True),
    Column("status", String, nullable=False, index=True),
    Column("input_artifact_refs", JSON, nullable=False),
    Column("input_artifact_sequence_refs", JSON, nullable=False),
    Column("output_artifact_refs", JSON, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("error", String, nullable=True),
    Column("queued_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
)

experiments = Table(
    "experiments",
    metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("description", String, nullable=True),
    Column("workflow_version_id", String, nullable=False, index=True),
    Column("status", String, nullable=False, index=True),
    Column("parameters", JSON, nullable=False),
    Column("input_artifact_refs", JSON, nullable=False),
    Column("input_artifact_sequence_refs", JSON, nullable=False),
    Column("variants", JSON, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)

node_runs = Table(
    "node_runs",
    metadata,
    Column("id", String, primary_key=True),
    Column("workflow_run_id", String, nullable=False, index=True),
    Column("workflow_node_id", String, nullable=False),
    Column("operator_id", String, nullable=False),
    Column("operator_version", String, nullable=False),
    Column("status", String, nullable=False, index=True),
    Column("input_artifact_refs", JSON, nullable=False),
    Column("output_artifact_refs", JSON, nullable=False),
    Column("attempt_count", Integer, nullable=False),
    Column("max_attempts", Integer, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("error", String, nullable=True),
    Column("queued_at", DateTime(timezone=True), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("finished_at", DateTime(timezone=True), nullable=True),
)

artifacts = Table(
    "artifacts",
    metadata,
    Column("id", String, primary_key=True),
    Column("artifact_type", String, nullable=False, index=True),
    Column("schema_version", Integer, nullable=False),
    Column("workflow_run_id", String, nullable=True, index=True),
    Column("producer_node_run_id", String, nullable=True, index=True),
    Column("payload_ref", String, nullable=False),
    Column("producer_operator_id", String, nullable=True),
    Column("producer_operator_version", String, nullable=True),
    Column("input_artifact_ids", JSON, nullable=False),
    Column("content_hash", String, nullable=True),
    Column("preview_ref", String, nullable=True),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

artifact_sequences = Table(
    "artifact_sequences",
    metadata,
    Column("id", String, primary_key=True),
    Column("artifact_type", String, nullable=False, index=True),
    Column("item_refs", JSON, nullable=False),
    Column("schema_version", Integer, nullable=False),
    Column("ordered", Boolean, nullable=False),
    Column("index_key", String, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

input_assembly_traces = Table(
    "input_assembly_traces",
    metadata,
    Column("id", String, primary_key=True),
    Column("node_run_id", String, nullable=False, index=True),
    Column("selected_inputs", JSON, nullable=False),
    Column("omitted_inputs", JSON, nullable=False),
    Column("policies", JSON, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

invocation_traces = Table(
    "invocation_traces",
    metadata,
    Column("id", String, primary_key=True),
    Column("node_run_id", String, nullable=False, index=True),
    Column("invocation_type", String, nullable=False),
    Column("input_artifact_refs", JSON, nullable=False),
    Column("output_artifact_refs", JSON, nullable=False),
    Column("provider", String, nullable=True),
    Column("model", String, nullable=True),
    Column("request_ref", String, nullable=True),
    Column("response_ref", String, nullable=True),
    Column("runtime", JSON, nullable=False),
    Column("metadata", JSON, nullable=False),
    Column("error", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

outbox_messages = Table(
    "outbox_messages",
    metadata,
    Column("id", String, primary_key=True),
    Column("subject", String, nullable=False, index=True),
    Column("message_type", String, nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    Column("status", String, nullable=False, index=True),
    Column("attempts", Integer, nullable=False),
    Column("error", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("published_at", DateTime(timezone=True), nullable=True),
)
