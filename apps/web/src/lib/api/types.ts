// TypeScript mirrors of the Notarius Studio FastAPI response schemas.
// UUIDs and datetimes arrive as strings over JSON.

export type UUID = string;
export type ISODateTime = string;

export type WorkflowRunStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export type NodeRunStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed_retryable"
  | "failed_permanent"
  | "cancelled";

export type ExecutionMode =
  | "single"
  | "map"
  | "reduce"
  | "stateful_sequence";

export interface Project {
  id: UUID;
  name: string;
  description: string | null;
  created_at: ISODateTime;
}

export interface Source {
  id: UUID;
  project_id: UUID;
  name: string;
  description: string | null;
  created_at: ISODateTime;
}

export interface SourceItem {
  id: UUID;
  source_id: UUID;
  order: number;
  text: string | null;
  image_path: string | null;
  metadata: Record<string, unknown>;
  created_at: ISODateTime;
}

export interface ArtifactRef {
  artifact_id: UUID;
  artifact_type: string;
  schema_version: number;
  content_hash: string | null;
}

export interface Artifact {
  id: UUID;
  artifact_type: string;
  schema_version: number;
  workflow_run_id: UUID | null;
  producer_node_run_id: UUID | null;
  payload_ref: string;
  producer_operator_id: string | null;
  producer_operator_version: string | null;
  input_artifact_ids: UUID[];
  content_hash: string | null;
  preview_ref: string | null;
  metadata: Record<string, unknown>;
  created_at: ISODateTime;
}

export interface ArtifactSequence {
  id: UUID;
  artifact_type: string;
  schema_version: number;
  item_refs: ArtifactRef[];
  ordered: boolean;
  index_key: string;
  metadata: Record<string, unknown>;
  created_at: ISODateTime;
}

export interface PortSpec {
  name: string;
  artifact_type: string;
  schema_version: number;
  sequence: boolean;
  required: boolean;
  description: string | null;
}

export interface NodeSpec {
  id: string;
  version: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  execution_mode: ExecutionMode;
  config_schema: Record<string, unknown>;
  display_name: string | null;
  description: string | null;
}

export interface ArtifactTypePortUse {
  operator_id: string;
  operator_version: string;
  port_name: string;
  sequence: boolean;
  required: boolean;
}

export interface ArtifactType {
  artifact_type: string;
  schema_version: number;
  sequence: boolean;
  consumed_by: ArtifactTypePortUse[];
  produced_by: ArtifactTypePortUse[];
}

export interface WorkflowTemplate {
  id: string;
  version: string;
  display_name: string;
  description: string;
  config_schema: Record<string, unknown>;
}

export interface WorkflowRun {
  id: UUID;
  workflow_version_id: UUID;
  status: WorkflowRunStatus;
  input_artifact_refs: ArtifactRef[];
  input_artifact_sequence_refs: ArtifactRef[];
  output_artifact_refs: ArtifactRef[];
  metadata: Record<string, unknown>;
  error: string | null;
  queued_at: ISODateTime;
  started_at: ISODateTime | null;
  finished_at: ISODateTime | null;
}

export interface NodeRun {
  id: UUID;
  workflow_run_id: UUID;
  workflow_node_id: string;
  operator_id: string;
  operator_version: string;
  status: NodeRunStatus;
  input_artifact_refs: Record<string, unknown>;
  output_artifact_refs: Record<string, unknown>;
  attempt_count: number;
  max_attempts: number;
  error: string | null;
  queued_at: ISODateTime;
  started_at: ISODateTime | null;
  finished_at: ISODateTime | null;
}

export interface WorkflowRunSummaryError {
  node_run_id: UUID | null;
  status: string;
  error: string;
}

export interface WorkflowRunSummary {
  workflow_run: WorkflowRun;
  node_runs: NodeRun[];
  artifacts: Artifact[];
  node_run_status_counts: Record<string, number>;
  artifact_counts: Record<string, number>;
  errors: WorkflowRunSummaryError[];
}

export interface OutputArtifactPayload {
  content_type: string;
  byte_size: number;
  json_payload: unknown | null;
  text: string | null;
  error: string | null;
}

export interface OutputArtifact {
  artifact: Artifact;
  payload: OutputArtifactPayload | null;
}

export interface WorkflowRunOutputBundle {
  workflow_run: WorkflowRun;
  artifacts: OutputArtifact[];
  artifact_sequences: ArtifactSequence[];
  traces: unknown[];
}

export interface ArtifactInspection {
  artifact: Artifact;
  payload: OutputArtifactPayload | null;
  lineage: unknown | null;
}

export interface WorkflowTemplateLaunchResponse {
  template: WorkflowTemplate;
  workflow_definition: unknown;
  workflow_version: { id: UUID };
  workflow_run: WorkflowRun;
  queued_node_run_ids: UUID[];
}

export interface WorkflowRunExecutionNodeError {
  node_run_id: UUID;
  error: string;
}

export interface WorkflowRunExecutionResponse {
  workflow_run_id: UUID;
  workflow_run: WorkflowRun;
  processed_node_run_ids: UUID[];
  errors: WorkflowRunExecutionNodeError[];
}

export interface ImageSourceUploadResponse {
  source: Source;
  items: SourceItem[];
  artifacts: Artifact[];
  sequence: ArtifactSequence;
}

export interface ArtifactSequenceRefInput {
  sequence_id: UUID;
  artifact_type: string;
  schema_version: number;
}

export interface LaunchTemplateInput {
  name?: string;
  description?: string;
  config: Record<string, unknown>;
  input_artifact_sequence_refs?: ArtifactSequenceRefInput[];
  metadata?: Record<string, unknown>;
  created_by?: string;
  change_note?: string;
}
