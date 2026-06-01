import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { LLMHealthWidget } from "@/components/dashboard/LLMHealthWidget";

vi.mock("@/utils/dateFormat", () => ({
  formatRelativeTime: (value: string) => `relative:${value}`,
}));

describe("LLMHealthWidget", () => {
  it("renders the disabled state when monitoring is not configured", () => {
    render(
      <LLMHealthWidget
        health={{
          enabled: false,
          healthy: false,
          endpoints: [],
        }}
      />,
    );

    expect(screen.getByText("LLM backend health")).toBeInTheDocument();
    expect(screen.getByText("Disabled")).toBeInTheDocument();
    expect(
      screen.getByText("LLM health monitoring is disabled for this deployment."),
    ).toBeInTheDocument();
    expect(screen.queryByText("No LLM endpoints are configured yet.")).not.toBeInTheDocument();
  });

  it("renders an open circuit alert with endpoint details", () => {
    render(
      <LLMHealthWidget
        health={{
          enabled: true,
          healthy: false,
          checked_at: "2026-04-08T10:06:00Z",
          endpoints: [
            {
              name: "litellm",
              healthy: false,
              circuit_state: "open",
              consecutive_failures: 5,
              last_error: "request failed: connection refused",
              last_success: "2026-04-08T09:58:00Z",
              last_checked: "2026-04-08T10:06:00Z",
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("LLM backend health")).toBeInTheDocument();
    expect(screen.getByText("Down")).toBeInTheDocument();
    expect(screen.getByText("Circuit breaker open")).toBeInTheDocument();
    expect(screen.getByText("litellm")).toBeInTheDocument();
    expect(screen.getByText("Open")).toBeInTheDocument();
    expect(screen.getByText("Unhealthy")).toBeInTheDocument();
    expect(screen.getByText(/5 failures/)).toBeInTheDocument();
    expect(screen.getByText("request failed: connection refused")).toBeInTheDocument();
    expect(screen.getAllByText(/relative:2026-04-08T10:06:00Z/)).toHaveLength(2);
    expect(screen.getByText(/Last health poll relative:2026-04-08T10:06:00Z/)).toBeInTheDocument();
  });
});
