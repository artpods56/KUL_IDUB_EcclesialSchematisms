"use client";

import * as React from "react";
import { useSWRConfig } from "swr";

import { deleteSession, getSession, oidcLoginUrl, safeReturnPath } from "@/lib/api/auth";
import { ApiError, onUnauthorized } from "@/lib/api/client";
import type { Session } from "@/lib/api/contract";

type AuthState =
  | { kind: "loading" }
  | { kind: "signed-out" }
  | { kind: "authenticated"; session: Session }
  | { kind: "expired" }
  | { kind: "unavailable" };

interface AuthSessionContextValue {
  session: Session;
  logout: () => Promise<void>;
}

const AuthSessionContext = React.createContext<AuthSessionContextValue | null>(
  null,
);

export function sessionFailureKind(error: unknown): "signed-out" | "unavailable" {
  return error instanceof ApiError && error.status === 401
    ? "signed-out"
    : "unavailable";
}

export async function clearProtectedSWRState(
  cache: { keys: () => IterableIterator<string>; delete: (key: string) => void },
): Promise<void> {
  for (const key of cache.keys()) cache.delete(key);
}

function readReturnPath(): string {
  if (typeof window === "undefined") return "/workspaces";
  return safeReturnPath(`${window.location.pathname}${window.location.search}`);
}

function openLogin(): void {
  if (typeof window === "undefined") return;
  window.location.assign(oidcLoginUrl(readReturnPath()));
}

function AuthFrame({
  state,
  onLogout,
  onRetry,
  children,
}: {
  state: AuthState;
  onLogout: () => Promise<void>;
  onRetry: () => void;
  children: React.ReactNode;
}) {
  if (state.kind === "loading") {
    return <AuthStatus title="Checking session" detail="Opening your workspace…" />;
  }
  if (state.kind === "signed-out") {
    return (
      <AuthStatus
        title="Sign in to Notarius"
        detail="Your workspaces and graphs are available after authentication."
        action={<button type="button" onClick={openLogin}>Continue with SSO</button>}
      />
    );
  }
  if (state.kind === "expired") {
    return (
      <AuthStatus
        title="Your session has expired"
        detail="Sign in again to return to the same Notarius surface."
        action={<button type="button" onClick={openLogin}>Sign in again</button>}
      />
    );
  }
  if (state.kind === "unavailable") {
    return (
      <AuthStatus
        title="Session service unavailable"
        detail="Notarius could not confirm your session. Check the connection and try again."
        action={<button type="button" onClick={onRetry}>Try again</button>}
      />
    );
  }

  return (
    <AuthSessionContext.Provider value={{ session: state.session, logout: onLogout }}>
      {children}
    </AuthSessionContext.Provider>
  );
}

function AuthStatus({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: React.ReactNode;
}) {
  return (
    <main className="ns-auth-threshold">
      <div className="ns-auth-threshold__mark" aria-hidden="true">N</div>
      <p className="ns-auth-threshold__eyebrow">NOTARIUS</p>
      <h1>{title}</h1>
      <p className="ns-auth-threshold__detail">{detail}</p>
      {action ? <div className="ns-auth-threshold__action">{action}</div> : null}
    </main>
  );
}

export function useAuthSession(): AuthSessionContextValue {
  const context = React.useContext(AuthSessionContext);
  if (!context) throw new Error("useAuthSession must be used inside AuthSessionBoundary");
  return context;
}

export function AuthSessionBoundary({ children }: { children: React.ReactNode }) {
  const { cache } = useSWRConfig();
  const [state, setState] = React.useState<AuthState>({ kind: "loading" });

  const clearState = React.useCallback(async () => {
    await clearProtectedSWRState(cache);
  }, [cache]);

  const expireSession = React.useCallback(() => {
    void clearState();
    setState((current) => current.kind === "authenticated" ? { kind: "expired" } : current);
  }, [clearState]);

  React.useEffect(() => onUnauthorized(expireSession), [expireSession]);

  const loadSession = React.useCallback((signal: AbortSignal) => {
    void getSession(signal)
      .then((session) => setState({ kind: "authenticated", session }))
      .catch((error: unknown) => {
        if (signal.aborted) return;
        setState({ kind: sessionFailureKind(error) });
      });
  }, []);

  React.useEffect(() => {
    const controller = new AbortController();
    loadSession(controller.signal);
    return () => controller.abort();
  }, [loadSession]);

  const logout = React.useCallback(async () => {
    try {
      await deleteSession();
    } finally {
      await clearState();
      setState({ kind: "signed-out" });
    }
  }, [clearState]);

  return (
    <AuthFrame
      state={state}
      onLogout={logout}
      onRetry={() => {
        setState({ kind: "loading" });
        loadSession(new AbortController().signal);
      }}
    >
      {state.kind === "authenticated" ? children : null}
    </AuthFrame>
  );
}
