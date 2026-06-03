import React from "react";
import { act, render, renderHook, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const apiFns = vi.hoisted(() => ({
  setGlobalApiKey: vi.fn<(key: string | null) => void>(),
  setGlobalAdminToken: vi.fn<(token: string | null) => void>(),
}));

vi.mock("@/services/api", () => ({
  setGlobalApiKey: apiFns.setGlobalApiKey,
  setGlobalAdminToken: apiFns.setGlobalAdminToken,
}));

function encrypt(value: string) {
  return btoa(value.split("").reverse().join(""));
}

async function loadAuthModule() {
  vi.resetModules();
  return import("@/contexts/AuthContext");
}

describe("AuthContext", () => {
  beforeEach(() => {
    localStorage.clear();
    apiFns.setGlobalApiKey.mockReset();
    apiFns.setGlobalAdminToken.mockReset();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("throws when useAuth is called outside the provider", async () => {
    const { useAuth } = await loadAuthModule();

    expect(() => renderHook(() => useAuth())).toThrow("useAuth must be used within AuthProvider");
  });

  it("loads valid stored credentials, authenticates, and exposes children", async () => {
    localStorage.setItem("af_api_key", encrypt("stored-api-key"));
    localStorage.setItem("af_admin_token", encrypt("stored-admin-token"));
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      status: 200,
    } as Response);

    const { AuthProvider, useAuth } = await loadAuthModule();

    function Consumer() {
      const auth = useAuth();
      return (
        <div>
          <span data-testid="api-key">{auth.apiKey}</span>
          <span data-testid="admin-token">{auth.adminToken}</span>
          <span data-testid="auth-required">{String(auth.authRequired)}</span>
          <span data-testid="is-authenticated">{String(auth.isAuthenticated)}</span>
        </div>
      );
    }

    render(
      <AuthProvider>
        <Consumer />
      </AuthProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("api-key")).toHaveTextContent("stored-api-key");
    });

    expect(screen.getByTestId("admin-token")).toHaveTextContent("stored-admin-token");
    expect(screen.getByTestId("auth-required")).toHaveTextContent("true");
    expect(screen.getByTestId("is-authenticated")).toHaveTextContent("true");
    expect(fetch).toHaveBeenCalledWith("/api/ui/v1/dashboard/summary", {
      headers: { "X-API-Key": "stored-api-key" },
    });
    expect(apiFns.setGlobalApiKey).toHaveBeenCalledWith("stored-api-key");
    expect(apiFns.setGlobalAdminToken).toHaveBeenCalledWith("stored-admin-token");
  });

  it("treats an open server as unauthenticated-but-allowed when no key is stored", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      status: 200,
    } as Response);

    const { AuthProvider, useAuth } = await loadAuthModule();

    function Consumer() {
      const auth = useAuth();
      return (
        <div>
          <span data-testid="auth-required">{String(auth.authRequired)}</span>
          <span data-testid="is-authenticated">{String(auth.isAuthenticated)}</span>
        </div>
      );
    }

    render(
      <AuthProvider>
        <Consumer />
      </AuthProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("auth-required")).toHaveTextContent("false");
    });

    expect(screen.getByTestId("is-authenticated")).toHaveTextContent("true");
  });

  it("clears invalid stored keys when the server returns 401", async () => {
    localStorage.setItem("af_api_key", "%%%invalid%%%");
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 401,
    } as Response);

    const { AuthProvider, useAuth } = await loadAuthModule();

    function Consumer() {
      const auth = useAuth();
      return (
        <div>
          <span data-testid="auth-required">{String(auth.authRequired)}</span>
          <span data-testid="is-authenticated">{String(auth.isAuthenticated)}</span>
          <span data-testid="api-key">{String(auth.apiKey)}</span>
        </div>
      );
    }

    render(
      <AuthProvider>
        <Consumer />
      </AuthProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId("auth-required")).toHaveTextContent("true");
    });

    expect(screen.getByTestId("is-authenticated")).toHaveTextContent("false");
    expect(screen.getByTestId("api-key")).toHaveTextContent("null");
    expect(localStorage.getItem("af_api_key")).toBeNull();
    expect(apiFns.setGlobalApiKey).toHaveBeenCalledWith(null);
  });

  it("supports setting and clearing auth state through the context API", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      status: 200,
    } as Response);

    const { AuthProvider, useAuth } = await loadAuthModule();

    const wrapper = ({ children }: React.PropsWithChildren) => <AuthProvider>{children}</AuthProvider>;
    const { result } = renderHook(() => useAuth(), { wrapper });

    await waitFor(() => {
      expect(result.current.isAuthenticated).toBe(true);
    });

    act(() => {
      result.current.setApiKey("fresh-key");
      result.current.setAdminToken("fresh-admin");
    });

    expect(result.current.apiKey).toBe("fresh-key");
    expect(result.current.adminToken).toBe("fresh-admin");
    expect(localStorage.getItem("af_api_key")).toBe(encrypt("fresh-key"));
    expect(localStorage.getItem("af_admin_token")).toBe(encrypt("fresh-admin"));

    act(() => {
      result.current.clearAuth();
    });

    expect(result.current.apiKey).toBeNull();
    expect(result.current.adminToken).toBeNull();
    expect(localStorage.getItem("af_api_key")).toBeNull();
    expect(localStorage.getItem("af_admin_token")).toBeNull();
    expect(apiFns.setGlobalApiKey).toHaveBeenLastCalledWith(null);
    expect(apiFns.setGlobalAdminToken).toHaveBeenLastCalledWith(null);
  });

  it("shows the loading state before the auth check resolves and falls back to required auth on network failure", async () => {
    let rejectRequest: ((reason?: unknown) => void) | undefined;
    vi.mocked(fetch).mockImplementation(
      () =>
        new Promise((_, reject) => {
          rejectRequest = reject;
        }) as Promise<Response>,
    );
    vi.spyOn(console, "error").mockImplementation(() => {});

    const { AuthProvider, useAuth } = await loadAuthModule();

    function Consumer() {
      const auth = useAuth();
      return <span data-testid="auth-required">{String(auth.authRequired)}</span>;
    }

    render(
      <AuthProvider>
        <Consumer />
      </AuthProvider>,
    );

    expect(screen.getByText("Connecting…")).toBeInTheDocument();

    act(() => {
      rejectRequest?.(new Error("offline"));
    });

    await waitFor(() => {
      expect(screen.getByTestId("auth-required")).toHaveTextContent("true");
    });
  });
});
