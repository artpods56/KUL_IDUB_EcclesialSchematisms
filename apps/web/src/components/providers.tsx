"use client";

import * as React from "react";
import { SWRConfig } from "swr";
import { ApiError, request } from "@/lib/api/client";
import { ThemeProvider } from "@/components/theme";

const apiFetcher = (path: string) => request<unknown>("GET", path);

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider>
      <SWRConfig
        value={{
          fetcher: apiFetcher,
          revalidateOnFocus: false,
          shouldRetryOnError: (err) => !(err instanceof ApiError && err.status === 404),
          errorRetryCount: 2,
        }}
      >
        {children}
      </SWRConfig>
    </ThemeProvider>
  );
}
