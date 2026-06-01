import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AgentsPage } from "@/pages/AgentsPage";
import type { AgentTagSummary } from "@/services/tagApprovalApi";
import type { AgentNodeSummary, ReasonerDefinition, SkillDefinition } from "@/types/agentfield";

type NodeDetails = {
  reasoners: ReasonerDefinition[];
  skills: SkillDefinition[];
};

const pageState = vi.hoisted(() => ({
  navigate: vi.fn<(path: string) => void>(),
  startAgent: vi.fn<(nodeId: string) => Promise<unknown>>(),
  nodes: [] as AgentNodeSummary[],
  tags: [] as AgentTagSummary[],
  isLoading: false,
  isError: false,
  error: null as Error | null,
  nodeDetailsById: {} as Record<string, NodeDetails>,
  nodeDetailsLoading: {} as Record<string, boolean>,
  nodeDetailsErrors: {} as Record<string, Error>,
}));

vi.mock("@/utils/dateFormat", () => ({
  formatCompactRelativeTime: () => "just now",
}));

vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...props }: React.PropsWithChildren<{ to: string } & React.AnchorHTMLAttributes<HTMLAnchorElement>>) => (
    <a href={to} {...props}>
      {children}
    </a>
  ),
  useNavigate: () => pageState.navigate,
}));

vi.mock("@/hooks/queries", () => ({
  useAgents: () => ({
    data: pageState.isError ? undefined : { nodes: pageState.nodes },
    isLoading: pageState.isLoading,
    isError: pageState.isError,
    error: pageState.error,
  }),
  useAgentTagSummaries: () => ({
    data: pageState.tags,
  }),
}));

vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey }: { queryKey: unknown[] }) => {
    const nodeId = String(queryKey[1]);
    const error = pageState.nodeDetailsErrors[nodeId];

    if (pageState.nodeDetailsLoading[nodeId]) {
      return {
        data: undefined,
        isLoading: true,
        isError: false,
        error: null,
      };
    }

    if (error) {
      return {
        data: undefined,
        isLoading: false,
        isError: true,
        error,
      };
    }

    return {
      data: pageState.nodeDetailsById[nodeId] ?? { reasoners: [], skills: [] },
      isLoading: false,
      isError: false,
      error: null,
    };
  },
}));

vi.mock("@/services/api", () => ({
  getNodeDetails: vi.fn(),
}));

vi.mock("@/services/configurationApi", () => ({
  startAgent: (nodeId: string) => pageState.startAgent(nodeId),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <section {...props}>{children}</section>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({
    children,
    showIcon: _showIcon,
    variant: _variant,
    size: _size,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLSpanElement> & {
    showIcon?: boolean;
    variant?: string;
    size?: string;
  }>) => <span {...props}>{children}</span>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    variant: _variant,
    size: _size,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: string;
    size?: string;
  }>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/input", () => ({
  Input: React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
    (props, ref) => <input ref={ref} {...props} />,
  ),
}));

vi.mock("@/components/ui/tabs", async () => {
  const ReactModule = await import("react");
  const TabsContext = ReactModule.createContext<{
    value: string;
    onValueChange?: (value: string) => void;
  }>({ value: "" });

  return {
    Tabs: ({
      children,
      value,
      onValueChange,
      ...props
    }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement> & {
      value: string;
      onValueChange?: (value: string) => void;
    }>) => (
      <TabsContext.Provider value={{ value, onValueChange }}>
        <div {...props}>{children}</div>
      </TabsContext.Provider>
    ),
    TabsList: ({
      children,
      variant: _variant,
      density: _density,
      ...props
    }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement> & {
      variant?: string;
      density?: string;
    }>) => <div {...props}>{children}</div>,
    TabsTrigger: ({
      children,
      value,
      variant: _variant,
      size: _size,
      onClick,
      ...props
    }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement> & {
      value: string;
      variant?: string;
      size?: string;
    }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return (
        <button
          type="button"
          aria-pressed={ctx.value === value}
          onClick={(event) => {
            onClick?.(event);
            ctx.onValueChange?.(value);
          }}
          {...props}
        >
          {children}
        </button>
      );
    },
    TabsContent: ({
      children,
      value,
      ...props
    }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement> & {
      value: string;
    }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return ctx.value === value ? <div {...props}>{children}</div> : null;
    },
  };
});

vi.mock("@/components/ui/endpoint-kind-icon-box", () => ({
  EndpointKindIconBox: (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />, 
}));

vi.mock("@/components/ui/entity-tag", () => ({
  EntityTag: ({ children, tone: _tone, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLSpanElement> & { tone?: string }>) => (
    <span {...props}>{children}</span>
  ),
}));

vi.mock("@/components/nodes", () => ({
  NodeProcessLogsPanel: ({ nodeId }: { nodeId: string }) => <div>Process logs for {nodeId}</div>,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    AgentNodeIcon: Icon,
    ChevronRight: Icon,
    Play: Icon,
    ReasonerIcon: Icon,
    RefreshCw: Icon,
    Search: Icon,
    SkillIcon: Icon,
    Terminal: Icon,
  };
});

describe("AgentsPage", () => {
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleErrorSpy = vi.spyOn(console, "error").mockImplementation((message) => {
      const text = String(message);
      if (
        text.includes("cannot be a descendant of <button>") ||
        text.includes("cannot contain a nested <button>")
      ) {
        return;
      }
    });
    pageState.navigate.mockReset();
    pageState.startAgent.mockReset();
    pageState.startAgent.mockResolvedValue({ ok: true });
    pageState.nodes = [];
    pageState.tags = [];
    pageState.isLoading = false;
    pageState.isError = false;
    pageState.error = null;
    pageState.nodeDetailsById = {};
    pageState.nodeDetailsLoading = {};
    pageState.nodeDetailsErrors = {};
  });

  afterEach(() => {
    consoleErrorSpy.mockRestore();
    vi.clearAllMocks();
  });

  it("renders loading, error, and empty states", () => {
    pageState.isLoading = true;
    const loadingView = render(<AgentsPage />);
    expect(screen.getByText("Loading agents…")).toBeInTheDocument();
    loadingView.unmount();

    pageState.isLoading = false;
    pageState.isError = true;
    pageState.error = new Error("request failed");
    const errorView = render(<AgentsPage />);
    expect(screen.getByText(/Failed to load agents: request failed/)).toBeInTheDocument();
    errorView.unmount();

    pageState.isError = false;
    render(<AgentsPage />);
    expect(screen.getByText("No agent nodes found")).toBeInTheDocument();
  });

  it("filters nodes and exercises expansion, endpoints, logs, and restart actions", async () => {
    pageState.nodes = buildAgentNodes();
    pageState.tags = [
      {
        agent_id: "agent-alpha",
        approved_tags: ["ops", "prod"],
        proposed_tags: ["billing"],
        lifecycle_status: "ready",
        registered_at: "2026-04-07T00:00:00Z",
      },
    ];
    pageState.nodeDetailsById["agent-alpha"] = {
      reasoners: [
        { id: "reasoner.summarizer", name: "Summarizer", description: "Summaries" },
        { id: "reasoner.planner", name: "Planner", description: "Plans work" },
        { id: "reasoner.qa", name: "QA", description: "Validates output" },
        { id: "reasoner.audit", name: "Audit", description: "Audits flows" },
        { id: "reasoner.ops", name: "Ops", description: "Operational checks" },
        { id: "reasoner.docs", name: "Docs", description: "Docs sync" },
      ],
      skills: [
        { id: "skill.deploy", name: "Deploy", description: "Ship releases" },
        { id: "skill.rollback", name: "Rollback", description: "Undo releases" },
        { id: "skill.rotate", name: "Rotate", description: "Rotate keys" },
        { id: "skill.observe", name: "Observe", description: "Check telemetry" },
      ],
    };

    render(<AgentsPage />);

    expect(screen.getByText("2 agent nodes registered")).toBeInTheDocument();
    expect(screen.getByText("Access management")).toBeInTheDocument();
    expect(screen.getByText("ops")).toBeInTheDocument();
    expect(screen.getByText("billing")).toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: /offline/i })[0]);
    expect(screen.getByText("agent-beta")).toBeInTheDocument();
    expect(screen.queryByText("agent-alpha")).not.toBeInTheDocument();

    fireEvent.click(screen.getAllByRole("button", { name: /all/i })[0]);
    fireEvent.change(screen.getByLabelText("Search agent nodes"), {
      target: { value: "missing" },
    });
    expect(screen.getByText("No matching agents")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }));
    expect(screen.getByText("agent-alpha")).toBeInTheDocument();

    fireEvent.click(screen.getByText("agent-alpha").closest("button")!);
    expect(screen.getByText("Reasoners")).toBeInTheDocument();
    expect(screen.getByText("Skills")).toBeInTheDocument();

    const endpointFilter = screen.getByLabelText("Filter reasoners and skills");
    fireEvent.change(endpointFilter, { target: { value: "deploy" } });
    expect(screen.getByRole("button", { name: /Open skill Deploy in playground/i })).toBeInTheDocument();
    expect(screen.queryByText("Summarizer")).not.toBeInTheDocument();

    fireEvent.change(endpointFilter, { target: { value: "missing" } });
    expect(screen.getByText('No matches for "missing"')).toBeInTheDocument();

    fireEvent.change(endpointFilter, { target: { value: "summarizer" } });
    fireEvent.click(screen.getByRole("button", { name: /Open reasoner Summarizer in playground/i }));
    expect(pageState.navigate).toHaveBeenCalledWith("/playground/agent-alpha.reasoner.summarizer");

    fireEvent.click(screen.getByLabelText("Open process logs for agent-alpha"));
    expect(screen.getByText("Process logs for agent-alpha")).toBeInTheDocument();

    fireEvent.click(screen.getAllByLabelText("Restart agent")[0]);
    await waitFor(() => {
      expect(pageState.startAgent).toHaveBeenCalledWith("agent-alpha");
    });
  });

  it("shows endpoint loading and error states for expanded nodes", () => {
    pageState.nodes = [buildAgentNodes()[0], buildAgentNodes()[1]];
    pageState.nodeDetailsLoading["agent-alpha"] = true;
    pageState.nodeDetailsErrors["agent-beta"] = new Error("node unreachable");

    const { container } = render(<AgentsPage />);

    fireEvent.click(screen.getByText("agent-alpha").closest("button")!);
    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByText("agent-beta").closest("button")!);
    expect(screen.getByText(/Could not load endpoints: node unreachable/)).toBeInTheDocument();
  });
});

function buildAgentNodes(): AgentNodeSummary[] {
  return [
    {
      id: "agent-alpha",
      base_url: "https://alpha.example.com",
      version: "1.2.3",
      team_id: "team-one",
      health_status: "ready",
      lifecycle_status: "ready",
      last_heartbeat: "2026-04-07T10:00:00Z",
      reasoner_count: 6,
      skill_count: 4,
    },
    {
      id: "agent-beta",
      base_url: "https://beta.example.com",
      version: "2.0.0",
      team_id: "team-one",
      health_status: "offline",
      lifecycle_status: "offline",
      last_heartbeat: "2026-04-07T09:00:00Z",
      reasoner_count: 1,
      skill_count: 0,
    },
  ];
}
