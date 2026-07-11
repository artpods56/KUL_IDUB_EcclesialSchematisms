"use client";

import * as React from "react";
import { SWRConfig } from "swr";
import * as stylex from "@stylexjs/stylex";
import { TooltipProvider } from "@/components/ui/tooltip";
import { API_BASE, ApiError } from "@/lib/api";

export const apiFetcher = async (path: string) => {
  const res = await fetch(`${API_BASE}${path}`, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      detail = JSON.stringify(await res.json());
    } catch {
      /* keep default */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json();
};

const s = stylex.create({
  toaster: {
    position: "fixed",
    bottom: "16px",
    right: "16px",
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    zIndex: 100,
  },
  toast: {
    borderRadius: "10px",
    backgroundColor: "#1a2238",
    border: "1px solid #2f3c5c",
    color: "#e6e9f2",
    padding: "10px 14px",
    fontSize: "0.85rem",
    boxShadow: "0 10px 30px rgba(0,0,0,0.4)",
    maxWidth: "420px",
  },
  toastError: { borderColor: "rgba(248,113,113,0.5)" },
});

interface Toast {
  id: number;
  message: string;
  kind: "info" | "error";
}

interface ToastContextValue {
  toast: (message: string, kind?: "info" | "error") => void;
}

const ToastContext = React.createContext<ToastContextValue | null>(null);

export function useToast() {
  const ctx = React.useContext(ToastContext);
  return ctx ?? { toast: () => {} };
}

export function Providers({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = React.useState<Toast[]>([]);
  const idRef = React.useRef(1);

  const toast = React.useCallback((message: string, kind: "info" | "error" = "info") => {
    const id = idRef.current++;
    setToasts((t) => [...t, { id, message, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5000);
  }, []);

  return (
    <SWRConfig
      value={{
        fetcher: apiFetcher,
        revalidateOnFocus: false,
        shouldRetryOnError: (err) => !(err instanceof ApiError && err.status === 404),
        errorRetryCount: 2,
      }}
    >
      <ToastContext.Provider value={{ toast }}>
        <TooltipProvider delay={200} timeout={100}>
          {children}
          <div {...stylex.props(s.toaster)}>
            {toasts.map((t) => (
              <div
                key={t.id}
                {...stylex.props(s.toast, t.kind === "error" ? s.toastError : null)}
              >
                {t.message}
              </div>
            ))}
          </div>
        </TooltipProvider>
      </ToastContext.Provider>
    </SWRConfig>
  );
}
