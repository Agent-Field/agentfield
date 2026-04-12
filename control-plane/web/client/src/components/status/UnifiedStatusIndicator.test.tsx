import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import LegacyStatusIndicator, {
  UnifiedStatusIndicator,
  getLifecycleStatusPriority,
  getStatusPriority,
  isStatusHealthy,
  statusNeedsAttention,
} from "@/components/status/UnifiedStatusIndicator";
import type { AgentStatus } from "@/types/agentfield";

describe("UnifiedStatusIndicator", () => {
  it("renders status details and transition metadata", () => {
    render(
      <UnifiedStatusIndicator
        status={
          {
            status: "ok",
            state: "starting",
            health_score: 55,
            last_seen: "2026-04-08T09:00:00Z",
            state_transition: {
              from: "inactive",
              to: "active",
              reason: "boot",
            },
          } as AgentStatus
        }
        showDetails
        size="lg"
      />,
    );

    expect(screen.getByText("Starting")).toBeInTheDocument();
    expect(screen.getByText("Health: 55%")).toBeInTheDocument();
    expect(screen.getByText(/Last seen:/)).toBeInTheDocument();
    expect(screen.getByText(/Transitioning: inactive/i)).toBeInTheDocument();
    expect(screen.getByText("Reason: boot")).toBeInTheDocument();
  });

  it("renders a title summary when details are hidden and falls back for invalid timestamps", () => {
    render(
      <UnifiedStatusIndicator
        status={
          {
            status: "ok",
            state: "active",
            health_score: 95,
            last_seen: "not-a-date",
          } as AgentStatus
        }
      />,
    );

    const summary = screen.getByTitle(
      "Agent is running and healthy - Health: 95% - Last seen: Unknown",
    );
    expect(summary).toBeInTheDocument();
    expect(screen.getByText("Active")).toBeInTheDocument();
  });
});

describe("UnifiedStatusIndicator helpers", () => {
  it("reports health, state, and lifecycle priorities", () => {
    expect(isStatusHealthy({ state: "active", health_score: 70 } as AgentStatus)).toBe(true);
    expect(isStatusHealthy({ state: "active", health_score: 69 } as AgentStatus)).toBe(false);
    expect(getStatusPriority("error")).toBe(0);
    expect(getStatusPriority("inactive")).toBe(4);
    expect(getLifecycleStatusPriority("ready")).toBe(3);
    expect(getLifecycleStatusPriority("ready", "inactive")).toBe(0);
    expect(statusNeedsAttention("degraded")).toBe(true);
    expect(statusNeedsAttention("ready")).toBe(false);
  });
});

describe("LegacyStatusIndicator", () => {
  it("renders normalized execution statuses with animation", () => {
    const { container } = render(<LegacyStatusIndicator status="running" />);

    expect(screen.getByText("Running")).toBeInTheDocument();
    expect(container.querySelector(".animate-ping")).not.toBeNull();
  });

  it("prefers inactive health status and can hide the label", () => {
    const { container } = render(
      <LegacyStatusIndicator
        status="ready"
        healthStatus="inactive"
        showLabel={false}
        className="custom-status"
      />,
    );

    expect(screen.queryByText("Offline")).not.toBeInTheDocument();
    expect(container.querySelector(".custom-status")).not.toBeNull();
    expect(container.querySelector(".bg-gray-400")).not.toBeNull();
  });
});
