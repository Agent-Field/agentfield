import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { HealthStrip } from "@/components/HealthStrip";

const state = vi.hoisted(() => ({
  llmHealth: vi.fn(),
  queueStatus: vi.fn(),
  agents: vi.fn(),
  sseSync: vi.fn(),
  refreshAllLiveQueries: vi.fn(),
}));

vi.mock("@/hooks/queries", () => ({
  useLLMHealth: () => state.llmHealth(),
  useQueueStatus: () => state.queueStatus(),
  useAgents: () => state.agents(),
}));

vi.mock("@/hooks/useSSEQuerySync", () => ({
  useSSESync: () => state.sseSync(),
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

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/separator", () => ({
  Separator: () => <hr />,
}));

vi.mock("@/components/ui/popover", () => ({
  Popover: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  PopoverTrigger: ({ children }: React.PropsWithChildren) => <>{children}</>,
  PopoverContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  Tooltip: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <>{children}</>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

describe("HealthStrip", () => {
  beforeEach(() => {
    state.refreshAllLiveQueries.mockReset();
    state.llmHealth.mockReturnValue({
      isLoading: false,
      data: { healthy: true, endpoints: [{ healthy: true }] },
    });
    state.queueStatus.mockReturnValue({
      data: { agents: { a: { running: 2 }, b: { running: 1 } } },
    });
    state.agents.mockReturnValue({
      data: {
        count: 2,
        nodes: [
          { health_status: "ready", lifecycle_status: "running" },
          { health_status: "active", lifecycle_status: "ready" },
        ],
      },
    });
    state.sseSync.mockReturnValue({
      execConnected: true,
      reconnecting: false,
      refreshAllLiveQueries: state.refreshAllLiveQueries,
    });
  });

  it("renders healthy connected status details", () => {
    render(<HealthStrip />);

    expect(screen.getByText("Healthy")).toBeInTheDocument();
    expect(screen.getByText("2/2 online")).toBeInTheDocument();
    expect(screen.getByText("3 running")).toBeInTheDocument();
    expect(screen.getByText("Live")).toBeInTheDocument();
    expect(
      screen.getByRole("button", {
        name: /System status: LLM healthy, 2 of 2 agents online, 3 running, Live/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Refresh runs, agents, and dashboard data" }),
    ).not.toBeInTheDocument();
  });

  it("renders loading and disconnected states and refreshes when requested", async () => {
    const user = userEvent.setup();

    state.llmHealth.mockReturnValue({
      isLoading: true,
      data: undefined,
    });
    state.queueStatus.mockReturnValue({ data: { agents: {} } });
    state.agents.mockReturnValue({
      data: {
        count: 1,
        nodes: [{ health_status: "idle", lifecycle_status: "stopped" }],
      },
    });
    state.sseSync.mockReturnValue({
      execConnected: false,
      reconnecting: false,
      refreshAllLiveQueries: state.refreshAllLiveQueries,
    });

    render(<HealthStrip />);

    expect(screen.getByText("Unknown")).toBeInTheDocument();
    expect(screen.getAllByText("Checking LLM health…")).toHaveLength(2);
    expect(screen.getByText("0/1 online")).toBeInTheDocument();
    expect(screen.getByText("Disconnected")).toBeInTheDocument();
    expect(
      screen.getAllByText(/Execution stream down .* Refresh to resync/i),
    ).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: "Refresh runs, agents, and dashboard data" }));
    expect(state.refreshAllLiveQueries).toHaveBeenCalledTimes(1);
  });
});
