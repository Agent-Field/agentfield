import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AdminTokenPrompt } from "@/components/AdminTokenPrompt";

const authState = vi.hoisted(() => ({
  adminToken: null as string | null,
  setAdminToken: vi.fn<(token: string | null) => void>(),
}));

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => authState,
}));

vi.mock("@/components/authorization/HintIcon", () => ({
  HintIcon: ({
    children,
  }: React.PropsWithChildren<{ label: string }>) => <span>{children}</span>,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  AlertDescription: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
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

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

describe("AdminTokenPrompt", () => {
  beforeEach(() => {
    authState.adminToken = null;
    authState.setAdminToken.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("saves a trimmed admin token and notifies callers", async () => {
    const user = userEvent.setup();
    const onTokenSet = vi.fn();

    render(<AdminTokenPrompt onTokenSet={onTokenSet} />);

    const input = screen.getByPlaceholderText("Same value as on the server");
    await user.type(input, "  admin-secret  ");
    await user.click(screen.getByRole("button", { name: "Save in browser" }));

    expect(authState.setAdminToken).toHaveBeenCalledWith("admin-secret");
    expect(onTokenSet).toHaveBeenCalledTimes(1);
    expect(screen.getByPlaceholderText("Same value as on the server")).toHaveValue("");
  });

  it("shows saved-token controls and supports editing, cancelling, and clearing", async () => {
    const user = userEvent.setup();
    authState.adminToken = "stored-token";

    render(<AdminTokenPrompt />);

    expect(screen.getByText("Admin token saved in this browser")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Change" }));
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByText("Admin token saved in this browser")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(authState.setAdminToken).toHaveBeenCalledWith(null);
  });
});
