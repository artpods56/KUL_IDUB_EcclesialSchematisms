import { request } from "./client";
import type {
  UUID,
  WorkflowRunExecutionResponse,
  WorkflowRunOutputBundle,
  WorkflowRunSummary,
} from "./types";

export function executeWorkflowRun(
  workflowRunId: UUID,
  maxNodeRuns = 100,
): Promise<WorkflowRunExecutionResponse> {
  return request("POST", `/v1/workflow-runs/${workflowRunId}/execute`, {
    body: { max_node_runs: maxNodeRuns },
  });
}

export function getWorkflowRunSummary(workflowRunId: UUID) {
  return request<WorkflowRunSummary>(
    "GET",
    `/v1/workflow-runs/${workflowRunId}/summary`,
  );
}

export function getWorkflowRunOutputs(
  workflowRunId: UUID,
  options: {
    artifactType?: string;
    includePayloads?: boolean;
    includeTextPayloads?: boolean;
  } = {},
) {
  return request<WorkflowRunOutputBundle>(
    "GET",
    `/v1/workflow-runs/${workflowRunId}/outputs`,
    {
      query: {
        artifact_type: options.artifactType,
        include_payloads: options.includePayloads ?? false,
        include_text_payloads: options.includeTextPayloads ?? false,
      },
    },
  );
}
