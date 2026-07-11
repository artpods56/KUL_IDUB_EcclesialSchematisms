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
  signal?: AbortSignal;
}

export async function request<T>(
  method: string,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, signal } = options;
  const headers: Record<string, string> = { Accept: "application/json" };
  const init: RequestInit = { method, headers, signal };
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }

  const res = await fetch(`${API_BASE}${path}`, init);
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
