"use client";

import * as React from "react";
import { SWRConfig, type Cache, type State } from "swr";

import { deleteSession, getSession, oidcLoginUrl, safeReturnPath } from "@/lib/api/auth";
import { ApiError, onUnauthorized } from "@/lib/api/client";
import type { Session } from "@/lib/api/contract";

type AuthState =
  | { kind: "loading" }
  | { kind: "signed-out" }
  | { kind: "authenticated"; session: Session }
  | { kind: "expired" }
  | { kind: "unavailable" }
  | { kind: "logout-failed" };

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

export function createProtectedSWRCache(): Cache<unknown> {
  return new Map<string, State<unknown>>();
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
  cacheGeneration,
  children,
}: {
  state: AuthState;
  onLogout: () => Promise<void>;
  onRetry: () => void;
  cacheGeneration: number;
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
  if (state.kind === "logout-failed") {
    return (
      <AuthStatus
        title="Sign out could not be completed"
        detail="The server could not revoke this session. Try again before leaving Notarius."
        action={<button type="button" onClick={onLogout}>Try sign out again</button>}
      />
    );
  }

  return (
    <SWRConfig
      key={`protected-cache-${cacheGeneration}-${state.session.id}-${state.session.user_id}`}
      value={{ provider: createProtectedSWRCache }}
    >
      <AuthSessionContext.Provider value={{ session: state.session, logout: onLogout }}>
        {children}
      </AuthSessionContext.Provider>
    </SWRConfig>
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
  const [state, setState] = React.useState<AuthState>({ kind: "loading" });
  const [cacheGeneration, setCacheGeneration] = React.useState(0);

  const expireSession = React.useCallback(() => {
    setState((current) => current.kind === "authenticated" ? { kind: "expired" } : current);
  }, []);

  React.useEffect(() => onUnauthorized(expireSession), [expireSession]);

  const loadSession = React.useCallback((signal: AbortSignal) => {
    void getSession(signal)
      .then((session) => {
        setCacheGeneration((current) => current + 1);
        setState({ kind: "authenticated", session });
      })
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
    setState({ kind: "logout-failed" });
    try {
      await deleteSession();
      setState({ kind: "signed-out" });
    } catch {
      setState({ kind: "logout-failed" });
    }
  }, []);

  return (
    <AuthFrame
      state={state}
      onLogout={logout}
      cacheGeneration={cacheGeneration}
      onRetry={() => {
        setState({ kind: "loading" });
        loadSession(new AbortController().signal);
      }}
    >
      {state.kind === "authenticated" ? children : null}
    </AuthFrame>
  );
}
