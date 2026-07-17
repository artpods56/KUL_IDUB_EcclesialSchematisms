import { API_BASE, request } from "./client";
import type {
  AppliedNodeSecret,
  ApplyNodeSecretRequest,
  CreateSavedGraphRequest,
  CreateSavedGraphResponse,
  GraphNodeSecrets,
  GraphMaterializations,
  RunRequest,
  RunResponse,
  SavedGraph,
  UploadRequest,
  UploadResponse,
  UpdateSavedGraphRequest,
} from "./contract";

export function getSavedGraph(
  graphId: string,
  signal?: AbortSignal,
) {
  return request<SavedGraph>(
    "GET",
    `/v1/graphs/${encodeURIComponent(graphId)}`,
    { signal },
  );
}

export function getGraphMaterializations(
  graphId: string,
  graphRevision: number,
  signal?: AbortSignal,
) {
  const query = new URLSearchParams({
    graph_revision: String(graphRevision),
  });
  return request<GraphMaterializations>(
    "GET",
    `/v1/graphs/${encodeURIComponent(graphId)}/materializations?${query}`,
    { signal },
  );
}

export function getGraphNodeSecrets(
  graphId: string,
  signal?: AbortSignal,
) {
  return request<GraphNodeSecrets>(
    "GET",
    `/v1/graphs/${encodeURIComponent(graphId)}/node-secrets`,
    { signal },
  );
}

export function applyNodeSecret(
  graphId: string,
  nodeId: string,
  name: string,
  requestBody: ApplyNodeSecretRequest,
) {
  return request<AppliedNodeSecret>(
    "PUT",
    `/v1/graphs/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/secrets/${encodeURIComponent(name)}`,
    { body: requestBody },
  );
}

export function removeNodeSecret(
  graphId: string,
  nodeId: string,
  name: string,
  expectedGraphRevision: number,
) {
  const query = new URLSearchParams({
    expected_graph_revision: String(expectedGraphRevision),
  });
  return request<undefined>(
    "DELETE",
    `/v1/graphs/${encodeURIComponent(graphId)}/nodes/${encodeURIComponent(nodeId)}/secrets/${encodeURIComponent(name)}?${query}`,
  );
}

export function createSavedGraph(requestBody: CreateSavedGraphRequest) {
  return request<CreateSavedGraphResponse>("POST", "/v1/graphs", {
    body: requestBody,
  });
}

export function updateSavedGraph(
  graphId: string,
  requestBody: UpdateSavedGraphRequest,
) {
  return request<SavedGraph>(
    "PUT",
    `/v1/graphs/${encodeURIComponent(graphId)}`,
    { body: requestBody },
  );
}

export function deleteSavedGraph(
  graphId: string,
  expectedRevision: number,
) {
  const query = new URLSearchParams({
    expected_revision: String(expectedRevision),
  });
  return request<undefined>(
    "DELETE",
    `/v1/graphs/${encodeURIComponent(graphId)}?${query}`,
  );
}

export function uploadImage(
  filename: string,
  contentBase64: string,
) {
  const body: UploadRequest = {
    filename,
    content_base64: contentBase64,
  };
  return request<UploadResponse>("POST", "/v1/uploads", {
    body,
  });
}

export function runGraph(requestBody: RunRequest) {
  return request<RunResponse>("POST", "/v1/runs", {
    body: requestBody,
  });
}

export function artifactContentUrl(
  contentUrl: string | null | undefined,
): string | null {
  if (!contentUrl) return null;
  return new URL(contentUrl, `${API_BASE}/v1/`).toString();
}

export async function fileToBase64(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const chunkSize = 32_768;
  let binary = "";
  for (let index = 0; index < bytes.length; index += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(index, index + chunkSize));
  }
  return window.btoa(binary);
}
