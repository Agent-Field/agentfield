import { render, screen, fireEvent } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NodeCard } from "./NodeCard";
import type { AgentNodeSummary, HealthStatus, LifecycleStatus } from "@/types/agentfield";

interface MockDidStatus {
  has_did: boolean;
  did_status: "active" | "inactive" | "revoked";
  reasoner_count: number;
  skill_count: number;
  last_updated: string;
}

type MockControlState =
  | "stopped"
  | "starting"
  | "running"
  | "stopping"
  | "error"
  | "reconciling";

interface MockAgentControlButtonProps {
  agentId: string;
  currentState: MockControlState;
  onToggle: (action: "start" | "stop" | "reconcile") => Promise<void>;
}

const mocks = vi.hoisted(() => ({
  navigate: vi.fn(),
  startAgent: vi.fn(),
  stopAgent: vi.fn(),
  reconcileAgent: vi.fn(),
  didStatus: null as MockDidStatus | null,
}));

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return {
    ...actual,
    useNavigate: () => mocks.navigate,
  };
});

vi.mock("../services/configurationApi", () => ({
  startAgent: mocks.startAgent,
  stopAgent: mocks.stopAgent,
  reconcileAgent: mocks.reconcileAgent,
}));

vi.mock("../hooks/useDIDInfo", () => ({
  useDIDStatus: () => ({
    status: mocks.didStatus,
    loading: false,
    error: null,
    refetch: vi.fn(),
  }),
}));

vi.mock("./did/DIDStatusBadge", () => ({
  CompositeDIDStatus: ({
    status,
    reasonerCount,
    skillCount,
  }: {
    status: string;
    reasonerCount: number;
    skillCount: number;
  }) => <span>{`${status}:${reasonerCount + skillCount}`}</span>,
}));

vi.mock("./did/DIDDisplay", () => ({
  DIDIdentityBadge: ({ nodeId }: { nodeId: string }) => <span>{`did:${nodeId}`}</span>,
}));

vi.mock("./mcp/MCPHealthIndicator", () => ({
  MCPHealthDot: ({ status }: { status: string }) => <span>{`dot:${status}`}</span>,
  MCPHealthIndicator: ({ status }: { status: string }) => <span>{`mcp:${status}`}</span>,
}));

vi.mock("@/components/ui/AgentControlButton", () => ({
  AgentControlButton: ({
    agentId,
    currentState,
    onToggle,
  }: MockAgentControlButtonProps) => {
    const actionMap: Record<MockControlState, "start" | "stop" | "reconcile"> = {
      stopped: "start",
      starting: "start",
      running: "stop",
      stopping: "stop",
      error: "reconcile",
      reconciling: "reconcile",
    };

    return (
      <button
        type="button"
        aria-label={`Control ${agentId}`}
        onClick={() => {
          void onToggle(actionMap[currentState]);
        }}
      >
        Control
      </button>
    );
  },
}));

const createNodeSummary = (
  overrides: Partial<AgentNodeSummary> = {}
): AgentNodeSummary => ({
  id: "node-1",
  base_url: "https://example.test",
  version: "1.2.3",
  team_id: "team-alpha",
  health_status: "ready",
  lifecycle_status: "ready",
  last_heartbeat: "2026-03-29T11:59:00.000Z",
  deployment_type: "serverless",
  reasoner_count: 5,
  skill_count: 4,
  mcp_summary: {
    service_status: "ready",
    running_servers: 2,
    total_servers: 3,
    total_tools: 8,
    overall_health: 92,
    has_issues: true,
    capabilities_available: true,
  },
  ...overrides,
});

describe("NodeCard", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-29T12:00:00.000Z"));
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.clearAllMocks();
    mocks.didStatus = {
      has_did: true,
      did_status: "active",
      reasoner_count: 2,
      skill_count: 3,
      last_updated: "2026-03-29T11:58:30.000Z",
    };
  });

  afterEach(() => {
    vi.useRealTimers();
    mocks.didStatus = null;
    vi.restoreAllMocks();
  });

  it("renders node information and metadata", () => {
    render(<NodeCard nodeSummary={createNodeSummary()} searchQuery="node" />);

    expect(
      screen.getByRole("button", {
        name: /navigate to details for node node-1\. status: ready\./i,
      })
    ).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /node\s*-1/i })).toBeInTheDocument();
    expect(screen.getByText("Ready")).toBeInTheDocument();
    expect(screen.getByText("v1.2.3")).toBeInTheDocument();
    expect(screen.getByText("Serverless")).toBeInTheDocument();
    expect(screen.getByText("High capability")).toBeInTheDocument();
    expect(screen.getByText("Issues detected")).toBeInTheDocument();
    expect(screen.getByText("1m ago")).toBeInTheDocument();
    expect(screen.getByText("5 reasoners")).toBeInTheDocument();
    expect(screen.getByText("4 skills")).toBeInTheDocument();
    expect(screen.getByText("Team team-alpha")).toBeInTheDocument();
    expect(screen.getByText("active:5")).toBeInTheDocument();
    expect(screen.getByText("did:node-1")).toBeInTheDocument();
    expect(screen.getByText("dot:running")).toBeInTheDocument();
    expect(screen.getByText("mcp:running")).toBeInTheDocument();
  });

  it("navigates to the node detail page on click and keyboard activation", () => {
    render(<NodeCard nodeSummary={createNodeSummary()} />);

    const card = screen.getByRole("button", {
      name: /navigate to details for node node-1/i,
    });

    fireEvent.click(card);
    expect(mocks.navigate).toHaveBeenCalledWith("/nodes/node-1");

    fireEvent.keyDown(card, { key: "Enter" });
    fireEvent.keyDown(card, { key: " " });
    expect(mocks.navigate).toHaveBeenCalledTimes(3);
  });

  it.each([
    {
      name: "running node",
      lifecycleStatus: "ready" as LifecycleStatus,
      healthStatus: "ready" as HealthStatus,
      expectedLabel: "Ready",
      expectedAction: mocks.stopAgent,
      mcpSummary: createNodeSummary().mcp_summary,
    },
    {
      name: "offline node",
      lifecycleStatus: "stopped" as LifecycleStatus,
      healthStatus: "inactive" as HealthStatus,
      expectedLabel: "Offline",
      expectedAction: mocks.startAgent,
      mcpSummary: undefined,
    },
    {
      name: "degraded node",
      lifecycleStatus: "degraded" as LifecycleStatus,
      healthStatus: "ready" as HealthStatus,
      expectedLabel: "Degraded",
      expectedAction: mocks.reconcileAgent,
      mcpSummary: {
        service_status: "degraded",
        running_servers: 1,
        total_servers: 3,
        total_tools: 8,
        overall_health: 40,
        has_issues: true,
        capabilities_available: false,
      },
    },
  ])(
    "renders the correct status and handles the action button for a $name",
    ({ lifecycleStatus, healthStatus, expectedLabel, expectedAction, mcpSummary }) => {
      render(
        <NodeCard
          nodeSummary={createNodeSummary({
            lifecycle_status: lifecycleStatus,
            health_status: healthStatus,
            mcp_summary: mcpSummary,
          })}
        />
      );

      expect(screen.getByText(expectedLabel)).toBeInTheDocument();

      fireEvent.click(screen.getByRole("button", { name: "Control node-1" }));

      expect(expectedAction).toHaveBeenCalledWith("node-1");
      expect(mocks.navigate).not.toHaveBeenCalled();
    }
  );

  it("handles a missing heartbeat without crashing", () => {
    render(
      <NodeCard
        nodeSummary={createNodeSummary({
          last_heartbeat: undefined,
          mcp_summary: undefined,
          deployment_type: undefined,
          reasoner_count: 1,
          skill_count: 0,
        })}
      />
    );

    expect(screen.getByText("No heartbeat")).toBeInTheDocument();
  });
});
