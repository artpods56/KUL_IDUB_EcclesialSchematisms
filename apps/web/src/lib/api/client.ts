import type {
  ImageSourceUploadResponse,
  ISODateTime,
  UUID,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_NOTARIUS_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`API error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

interface RequestOptions {
  body?: unknown;
  query?: Record<string, string | number | boolean | null | undefined>;
  signal?: AbortSignal;
  fetchOptions?: RequestInit;
}

function buildUrl(path: string, query?: RequestOptions["query"]): string {
  const url = new URL(API_BASE + path);
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      if (value !== null && value !== undefined) {
        url.searchParams.set(key, String(value));
      }
    }
  }
  return url.toString();
}

export async function request<T>(
  method: string,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, query, signal, fetchOptions } = options;
  const headers: Record<string, string> = {
    Accept: "application/json",
    ...(fetchOptions?.headers as Record<string, string> | undefined),
  };
  const init: RequestInit = { method, headers, ...fetchOptions, signal };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  const res = await fetch(buildUrl(path, query), init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = typeof data === "string" ? data : (data.detail ?? JSON.stringify(data));
    } catch {
      try {
        detail = await res.text();
      } catch {
        /* keep default */
      }
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

/** Upload page images as a multipart source. Returns the page-image sequence. */
export async function uploadImageSource(
  projectId: UUID,
  files: File[],
  name?: string,
): Promise<ImageSourceUploadResponse> {
  const form = new FormData();
  if (name) form.append("name", name);
  for (const file of files) form.append("files", file, file.name);

  const res = await fetch(
    buildUrl(`/v1/projects/${projectId}/sources/images`),
    { method: "POST", body: form, headers: { Accept: "application/json" } },
  );
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      /* keep default */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as ImageSourceUploadResponse;
}

/** A browser-loadable URL for an artifact's raw payload (e.g. a page image). */
export function artifactPayloadUrl(artifactId: UUID): string {
  return buildUrl(`/v1/artifacts/${artifactId}/payload`);
}

export function formatDate(iso: ISODateTime | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}
