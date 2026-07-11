import { API_BASE, request } from "./client";
import type {
  RunRequest,
  RunResponse,
  UploadRequest,
  UploadResponse,
} from "./contract";

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
