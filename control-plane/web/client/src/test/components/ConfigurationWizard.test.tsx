import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConfigurationWizard } from "@/components/forms/ConfigurationWizard";
import type { AgentPackage, ConfigurationSchema } from "@/types/agentfield";

const wizardState = vi.hoisted(() => ({
  submittedConfig: { api_key: "secret-token", enabled: true, region: "" },
}));

vi.mock("@/components/forms/ConfigurationForm", () => ({
  ConfigurationForm: ({
    onSubmit,
  }: {
    schema: ConfigurationSchema;
    initialValues?: Record<string, unknown>;
    onSubmit: (config: Record<string, unknown>) => Promise<void>;
    title: string;
    description: string;
  }) => (
    <div>
      <div>Configuration form</div>
      <button type="button" onClick={() => void onSubmit(wizardState.submittedConfig)}>
        Submit configuration
      </button>
    </div>
  ),
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

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = () => <span>icon</span>;
  return {
    ChevronLeft: Icon,
    ChevronRight: Icon,
    Check: Icon,
    Settings: Icon,
    Package: Icon,
    Info: Icon,
  };
});

describe("ConfigurationWizard", () => {
  const pkg: AgentPackage = {
    id: "pkg-1",
    name: "Example package",
    version: "1.2.3",
    description: "Example description",
    author: "AgentField",
    tags: ["ops", "alpha"],
  };

  const schema: ConfigurationSchema = {
    fields: [
      { name: "api_key", type: "secret" },
      { name: "enabled", type: "boolean" },
      { name: "region", type: "text" },
    ],
  };

  beforeEach(() => {
    wizardState.submittedConfig = { api_key: "secret-token", enabled: true, region: "" };
  });

  it("walks through the wizard and completes with the submitted configuration", async () => {
    const onComplete = vi.fn().mockResolvedValue(undefined);

    render(
      <ConfigurationWizard
        package={pkg}
        schema={schema}
        onComplete={onComplete}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByRole("heading", { name: "Package Overview" })).toBeInTheDocument();
    expect(screen.getByText("Example package")).toBeInTheDocument();
    expect(screen.getByText("ops")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /next/i }));
    expect(screen.getByText("Configuration form")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Submit configuration" }));
    expect(await screen.findByText("Configuration Summary")).toBeInTheDocument();
    expect(screen.getByText("••••••••")).toBeInTheDocument();
    expect(screen.getByText("Enabled")).toBeInTheDocument();
    expect(screen.getByText("Not set")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Complete Setup" }));

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalledWith(wizardState.submittedConfig);
    });
  });

  it("uses the cancel action on the first step", () => {
    const onCancel = vi.fn();

    render(
      <ConfigurationWizard
        package={pkg}
        schema={schema}
        onComplete={vi.fn()}
        onCancel={onCancel}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
