import { API_BASE, request } from "./client";
import type {
  CreateSavedGraphRequest,
  CreateSavedGraphResponse,
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

export function uploadFile(
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
