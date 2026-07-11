import { API_BASE, request } from "./client";
import type {
  PrototypeNodeRegistry,
  PrototypeRunRequest,
  PrototypeRunResponse,
  PrototypeSampleRequest,
  PrototypeSampleResponse,
  PrototypeUploadRequest,
  PrototypeUploadResponse,
} from "./prototype-contract";

export function getPrototypeRegistry() {
  return request<PrototypeNodeRegistry>("GET", "/v1/prototype/nodes");
}

export function uploadPrototypeFile(
  filename: string,
  contentBase64: string,
) {
  const body: PrototypeUploadRequest = {
    filename,
    content_base64: contentBase64,
  };
  return request<PrototypeUploadResponse>("POST", "/v1/prototype/uploads", {
    body,
  });
}

export function createPrototypeSamples(count: number) {
  const body: PrototypeSampleRequest = { count };
  return request<PrototypeSampleResponse>("POST", "/v1/prototype/samples", {
    body,
  });
}

export function runPrototypeGraph(requestBody: PrototypeRunRequest) {
  return request<PrototypeRunResponse>("POST", "/v1/prototype/run", {
    body: requestBody,
  });
}

export function prototypeContentUrl(
  contentUrl: string | null | undefined,
): string | null {
  if (!contentUrl) return null;
  return new URL(contentUrl, `${API_BASE}/v1/prototype/`).toString();
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

export async function loadPrototypeArtifactContent(
  contentUrl: string,
): Promise<unknown> {
  const response = await fetch(contentUrl, {
    headers: { Accept: "application/json, text/plain, text/csv, */*" },
  });
  if (!response.ok) {
    throw new Error(`Could not load artifact content (${response.status})`);
  }

  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    return response.json() as Promise<unknown>;
  }
  return response.text();
}
