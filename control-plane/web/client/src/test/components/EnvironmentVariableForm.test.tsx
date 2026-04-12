import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EnvironmentVariableForm } from "@/components/forms/EnvironmentVariableForm";

const envState = vi.hoisted(() => ({
  getAgentEnvironmentVariables: vi.fn(),
  updateAgentEnvironmentVariables: vi.fn(),
  getAgentConfigurationSchema: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@/services/api", () => ({
  getAgentEnvironmentVariables: (...args: unknown[]) => envState.getAgentEnvironmentVariables(...args),
  updateAgentEnvironmentVariables: (...args: unknown[]) => envState.updateAgentEnvironmentVariables(...args),
  getAgentConfigurationSchema: (...args: unknown[]) => envState.getAgentConfigurationSchema(...args),
}));

vi.mock("@/components/ui/notification", () => ({
  useSuccessNotification: () => envState.success,
  useErrorNotification: () => envState.error,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children }: React.PropsWithChildren) => <section>{children}</section>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardDescription: ({ children }: React.PropsWithChildren) => <p>{children}</p>,
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <h2>{children}</h2>,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: () => <div>loading</div>,
}));

vi.mock("@/components/ui/input", () => ({
  Input: React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
    (props, ref) => <input ref={ref} {...props} />,
  ),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({
    children,
    ...props
  }: React.PropsWithChildren<React.LabelHTMLAttributes<HTMLLabelElement>>) => (
    <label {...props}>{children}</label>
  ),
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = () => <span>icon</span>;
  return {
    Loader2: Icon,
    Save: Icon,
    AlertCircle: Icon,
    Eye: Icon,
    EyeOff: Icon,
    RefreshCw: Icon,
  };
});

describe("EnvironmentVariableForm", () => {
  beforeEach(() => {
    envState.getAgentConfigurationSchema.mockReset();
    envState.getAgentEnvironmentVariables.mockReset();
    envState.updateAgentEnvironmentVariables.mockReset();
    envState.success.mockReset();
    envState.error.mockReset();

    envState.getAgentConfigurationSchema.mockResolvedValue({
      schema: {
        user_environment: {
          required: [
            {
              name: "API_KEY",
              type: "secret",
              required: true,
              description: "API secret",
              validation: { pattern: "^sk-.+" },
            },
          ],
          optional: [
            {
              name: "TIMEOUT",
              type: "number",
              description: "Timeout seconds",
              validation: { min: 1, max: 10 },
            },
          ],
        },
      },
      metadata: {
        package_name: "Example package",
        package_version: "2.0.0",
      },
    });
    envState.getAgentEnvironmentVariables.mockResolvedValue({
      variables: {
        API_KEY: "sk-old",
        TIMEOUT: "3",
      },
    });
    envState.updateAgentEnvironmentVariables.mockResolvedValue({});
  });

  it("loads environment data, toggles secrets, saves changes, and reloads", async () => {
    const onConfigurationChange = vi.fn();

    render(
      <EnvironmentVariableForm
        agentId="agent-1"
        packageId="pkg-1"
        onConfigurationChange={onConfigurationChange}
      />,
    );

    expect(await screen.findByText("Environment Variables")).toBeInTheDocument();
    const secretInput = screen.getByDisplayValue("sk-old");
    expect(secretInput).toHaveAttribute("type", "password");

    fireEvent.click(screen.getAllByRole("button")[1]);
    expect(screen.getByDisplayValue("sk-old")).toHaveAttribute("type", "text");

    fireEvent.change(screen.getByDisplayValue("sk-old"), {
      target: { value: "sk-new" },
    });
    fireEvent.change(screen.getByDisplayValue("3"), {
      target: { value: "8" },
    });

    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => {
      expect(envState.updateAgentEnvironmentVariables).toHaveBeenCalledWith("agent-1", "pkg-1", {
        API_KEY: "sk-new",
        TIMEOUT: "8",
      });
    });
    expect(envState.success).toHaveBeenCalledWith("Environment variables saved successfully");
    expect(onConfigurationChange).toHaveBeenCalledTimes(1);
    expect(envState.getAgentConfigurationSchema).toHaveBeenCalledTimes(2);
    expect(envState.getAgentEnvironmentVariables).toHaveBeenCalledTimes(2);
  });

  it("shows validation errors and blocks save when values are invalid", async () => {
    render(<EnvironmentVariableForm agentId="agent-1" packageId="pkg-1" />);

    expect(await screen.findByText("Required Variables")).toBeInTheDocument();

    fireEvent.change(screen.getByDisplayValue("sk-old"), {
      target: { value: "" },
    });
    fireEvent.change(screen.getByDisplayValue("3"), {
      target: { value: "99" },
    });
    fireEvent.click(screen.getByRole("button", { name: /save changes/i }));

    expect(await screen.findByText("API_KEY is required")).toBeInTheDocument();
    expect(screen.getByText("TIMEOUT must be at most 10")).toBeInTheDocument();
    expect(envState.updateAgentEnvironmentVariables).not.toHaveBeenCalled();
    expect(envState.error).toHaveBeenCalledWith("Please fix the validation errors before saving");
  });
});
