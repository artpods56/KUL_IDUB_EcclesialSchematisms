"use client";

import useSWR from "swr";
import type { PrototypeNodeRegistry } from "@/lib/api";

/** Keyed SWR hooks over the Notarius API (global fetcher is `apiFetcher`). */

export function usePrototypeRegistry() {
  return useSWR<PrototypeNodeRegistry>("/v1/prototype/nodes");
}
