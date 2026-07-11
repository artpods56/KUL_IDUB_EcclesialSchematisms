import { request } from "./client";
import type {
  ArtifactType,
  LaunchTemplateInput,
  NodeSpec,
  UUID,
  WorkflowTemplate,
  WorkflowTemplateLaunchResponse,
} from "./types";

export function listNodeSpecs() {
  return request<NodeSpec[]>("GET", "/v1/node-specs");
}

export function listArtifactTypes() {
  return request<ArtifactType[]>("GET", "/v1/artifact-types");
}

export function listWorkflowTemplates() {
  return request<WorkflowTemplate[]>("GET", "/v1/workflow-templates");
}

export function launchTemplate(
  templateId: string,
  input: LaunchTemplateInput,
) {
  return request<WorkflowTemplateLaunchResponse>(
    "POST",
    `/v1/workflow-templates/${templateId}/launch`,
    { body: input },
  );
}

export type { UUID, WorkflowTemplateLaunchResponse };
