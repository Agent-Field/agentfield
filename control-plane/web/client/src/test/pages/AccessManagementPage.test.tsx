import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AccessManagementPage } from "@/pages/AccessManagementPage";

const pageState = vi.hoisted(() => ({
  adminToken: "admin-token" as string | null,
  invalidateQueries: vi.fn(),
  probeQuery: {
    data: true,
    isLoading: false,
    isSuccess: true,
    isError: false,
    isFetching: false,
    error: null as Error | null,
  },
  policiesQuery: {
    data: [{ id: 1, name: "policy-a" }],
    isLoading: false,
    isFetching: false,
    isError: false,
    error: null as Error | null,
    refetch: vi.fn(),
  },
  tagsQuery: {
    data: [
      {
        agent_id: "agent-pending",
        proposed_tags: ["alpha"],
        approved_tags: [],
        lifecycle_status: "pending_approval",
        registered_at: "2026-04-01T00:00:00Z",
      },
    ],
    isLoading: false,
    isFetching: false,
    isError: false,
    error: null as Error | null,
  },
}));

vi.mock("@/contexts/AuthContext", () => ({
  useAuth: () => ({
    adminToken: pageState.adminToken,
  }),
}));

vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({
    invalidateQueries: pageState.invalidateQueries,
  }),
}));

vi.mock("@/hooks/queries", () => ({
  ACCESS_MANAGEMENT_QUERY_KEY: "access-management",
  useAccessAdminRoutesProbe: () => pageState.probeQuery,
  useAccessPolicies: () => pageState.policiesQuery,
  useAgentTagSummaries: () => pageState.tagsQuery,
}));

vi.mock("@/components/ui/notification", () => ({
  NotificationProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/components/AdminTokenPrompt", () => ({
  AdminTokenPrompt: ({ onTokenSet }: { onTokenSet: () => void }) => (
    <button type="button" onClick={onTokenSet}>
      Save token
    </button>
  ),
}));

vi.mock("@/components/authorization/AccessRulesTab", () => ({
  AccessRulesTab: ({
    policies,
    canMutate,
    fetchError,
    onRefresh,
  }: {
    policies: Array<{ id: number; name: string }>;
    canMutate: boolean;
    fetchError?: Error | null;
    onRefresh: () => void;
    loading: boolean;
  }) => (
    <div>
      <div>Rules count: {policies.length}</div>
      <div>Rules can mutate: {String(canMutate)}</div>
      <div>Rules fetch error: {fetchError?.message ?? "none"}</div>
      <button type="button" onClick={onRefresh}>
        Refresh policies
      </button>
    </div>
  ),
}));

vi.mock("@/components/authorization/AgentTagsTab", () => ({
  AgentTagsTab: ({
    agents,
    canMutate,
    agentsError,
    onRefresh,
  }: {
    agents: Array<{ agent_id: string }>;
    canMutate: boolean;
    agentsError?: Error | null;
    onRefresh: () => void;
    agentsLoading: boolean;
    policies: unknown[];
  }) => (
    <div>
      <div>Agents count: {agents.length}</div>
      <div>Agents can mutate: {String(canMutate)}</div>
      <div>Agents error: {agentsError?.message ?? "none"}</div>
      <button type="button" onClick={onRefresh}>
        Refresh agents
      </button>
    </div>
  ),
}));

vi.mock("@/components/authorization/HintIcon", () => ({
  HintIcon: ({ children }: React.PropsWithChildren<{ label: string }>) => <span>{children}</span>,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertTitle: ({ children }: React.PropsWithChildren) => <h3>{children}</h3>,
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
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
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <h2>{children}</h2>,
}));

vi.mock("@/components/ui/tabs", async () => {
  const ReactModule = await import("react");
  const TabsContext = ReactModule.createContext<{
    value: string;
    setValue: (value: string) => void;
  }>({ value: "access-rules", setValue: () => {} });

  return {
    Tabs: ({
      children,
      defaultValue,
    }: React.PropsWithChildren<{ defaultValue: string }>) => {
      const [value, setValue] = ReactModule.useState(defaultValue);
      return <TabsContext.Provider value={{ value, setValue }}>{children}</TabsContext.Provider>;
    },
    TabsList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    TabsTrigger: ({
      children,
      value,
    }: React.PropsWithChildren<{ value: string }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return (
        <button type="button" aria-pressed={ctx.value === value} onClick={() => ctx.setValue(value)}>
          {children}
        </button>
      );
    },
    TabsContent: ({
      children,
      value,
    }: React.PropsWithChildren<{ value: string }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return ctx.value === value ? <div>{children}</div> : null;
    },
  };
});

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: () => <div>Loading skeleton</div>,
}));

vi.mock("lucide-react", () => ({
  ExternalLink: () => <span>external-link</span>,
  RefreshCw: () => <span>refresh-icon</span>,
}));

describe("AccessManagementPage", () => {
  beforeEach(() => {
    pageState.adminToken = "admin-token";
    pageState.invalidateQueries.mockReset();
    pageState.policiesQuery.refetch.mockReset();
    pageState.probeQuery = {
      data: true,
      isLoading: false,
      isSuccess: true,
      isError: false,
      isFetching: false,
      error: null,
    };
    pageState.policiesQuery = {
      data: [{ id: 1, name: "policy-a" }],
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    };
    pageState.tagsQuery = {
      data: [
        {
          agent_id: "agent-pending",
          proposed_tags: ["alpha"],
          approved_tags: [],
          lifecycle_status: "pending_approval",
          registered_at: "2026-04-01T00:00:00Z",
        },
      ],
      isLoading: false,
      isFetching: false,
      isError: false,
      error: null,
    };
  });

  it("shows the disabled-server guidance when admin routes are unavailable", () => {
    pageState.probeQuery = {
      data: false,
      isLoading: false,
      isSuccess: true,
      isError: false,
      isFetching: false,
      error: null,
    };

    render(<AccessManagementPage />);

    expect(
      screen.getByText("Authorization APIs are not enabled on this server"),
    ).toBeInTheDocument();
    expect(screen.getByText("AGENTFIELD_AUTHORIZATION_ENABLED=true")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /knowledge base: tag-based authorization/i })).toHaveAttribute(
      "href",
      "/api/v1/agentic/kb/articles/identity/tag-authorization",
    );
    expect(screen.getByText("Rules can mutate: false")).toBeInTheDocument();
  });

  it("renders both tabs and invalidates queries from refresh actions", () => {
    render(<AccessManagementPage />);

    expect(screen.getByText("Access management")).toBeInTheDocument();
    expect(screen.getByText("Rules count: 1")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /refresh-icon refresh/i }));
    fireEvent.click(screen.getByRole("button", { name: "Save token" }));
    fireEvent.click(screen.getByRole("button", { name: "Refresh policies" }));

    fireEvent.click(screen.getByRole("button", { name: /agent tags/i }));
    expect(screen.getByText("Agents count: 1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Refresh agents" }));

    expect(pageState.invalidateQueries).toHaveBeenCalledTimes(3);
    expect(pageState.invalidateQueries).toHaveBeenCalledWith({
      queryKey: ["access-management"],
    });
    expect(pageState.policiesQuery.refetch).toHaveBeenCalledTimes(1);
  });
});
