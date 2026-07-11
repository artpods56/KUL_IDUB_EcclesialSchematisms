"use client";

import useSWR from "swr";
import type { NodeRegistry } from "@/lib/api";

/** Keyed SWR hooks over the Notarius API (global fetcher is `apiFetcher`). */

export function useNodeRegistry() {
  return useSWR<NodeRegistry>("/v1/nodes");
}
