import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AuthGuard } from "@/components/AuthGuard";

const authState = vi.hoisted(() => ({
  apiKey: null as string | null,
  isAuthenticated: false,
  authRequired: true,
  setApiKey: vi.fn<(key: string | null) => void>(),
  setAdminToken: vi.fn<(token: string | null) => void>(),
}));

const apiFns = vi.hoisted(() => ({
  setGlobalApiKey: vi.fn<(key: string | null) => void>(),
}));

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => authState,
}));

vi.mock("@/services/api", () => ({
  setGlobalApiKey: apiFns.setGlobalApiKey,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button {...props}>{children}</button>
  ),
}));

vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardContent: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div {...props}>{children}</div>,
  CardDescription: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLParagraphElement>>) => <p {...props}>{children}</p>,
  CardHeader: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => <div {...props}>{children}</div>,
  CardTitle: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => <h2 {...props}>{children}</h2>,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  AlertDescription: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLParagraphElement>>) => <p {...props}>{children}</p>,
}));

describe("AuthGuard", () => {
  beforeEach(() => {
    authState.apiKey = null;
    authState.isAuthenticated = false;
    authState.authRequired = true;
    authState.setApiKey.mockReset();
    authState.setAdminToken.mockReset();
    apiFns.setGlobalApiKey.mockReset();
    vi.stubGlobal("fetch", vi.fn());
  });

  it("renders children immediately when auth is not required", () => {
    authState.authRequired = false;
    authState.apiKey = "stored-key";

    render(
      <AuthGuard>
        <div>Protected content</div>
      </AuthGuard>,
    );

    expect(screen.getByText("Protected content")).toBeInTheDocument();
    expect(apiFns.setGlobalApiKey).toHaveBeenCalledWith("stored-key");
  });

  it("submits credentials, persists the admin token, and shows validation state", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
    } as Response);

    render(
      <AuthGuard>
        <div>Protected content</div>
      </AuthGuard>,
    );

    fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "hax_live_123" } });
    fireEvent.change(screen.getByLabelText(/Admin Token/i), { target: { value: "  root-token  " } });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    expect(screen.getByRole("button", { name: /Validating/i })).toBeDisabled();

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith("/api/ui/v1/dashboard/summary", {
        headers: { "X-API-Key": "hax_live_123" },
      });
    });

    expect(authState.setApiKey).toHaveBeenCalledWith("hax_live_123");
    expect(authState.setAdminToken).toHaveBeenCalledWith("root-token");
    expect(apiFns.setGlobalApiKey).toHaveBeenCalledWith("hax_live_123");
  });

  it("shows an invalid key error when the server rejects the request", async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
    } as Response);

    render(
      <AuthGuard>
        <div>Protected content</div>
      </AuthGuard>,
    );

    fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "bad-key" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    expect(await screen.findByText("Invalid API key. Check the key and try again.")).toBeInTheDocument();
    expect(authState.setApiKey).not.toHaveBeenCalled();
  });

  it("shows a network error when validation cannot reach the server", async () => {
    vi.mocked(fetch).mockRejectedValue(new Error("offline"));

    render(
      <AuthGuard>
        <div>Protected content</div>
      </AuthGuard>,
    );

    fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "key" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));

    expect(
      await screen.findByText("Unable to reach the control plane. Is the server running?"),
    ).toBeInTheDocument();
  });
});
