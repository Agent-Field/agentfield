import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AgentTagsTab } from "@/components/authorization/AgentTagsTab";
import type { AccessPolicy } from "@/services/accessPoliciesApi";
import type { AgentTagSummary } from "@/services/tagApprovalApi";

const tagsState = vi.hoisted(() => ({
  approveAgentTags: vi.fn(),
  rejectAgentTags: vi.fn(),
  revokeAgentTags: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@/services/tagApprovalApi", async () => {
  const actual = await vi.importActual<typeof import("@/services/tagApprovalApi")>(
    "@/services/tagApprovalApi",
  );
  return {
    ...actual,
    approveAgentTags: (...args: unknown[]) => tagsState.approveAgentTags(...args),
    rejectAgentTags: (...args: unknown[]) => tagsState.rejectAgentTags(...args),
    revokeAgentTags: (...args: unknown[]) => tagsState.revokeAgentTags(...args),
  };
});

vi.mock("@/components/ui/notification", () => ({
  useSuccessNotification: () => tagsState.success,
  useErrorNotification: () => tagsState.error,
}));

vi.mock("@/utils/dateFormat", () => ({
  formatRelativeTime: () => "recently",
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

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/SearchBar", () => ({
  SearchBar: ({
    value,
    onChange,
    placeholder,
  }: {
    value: string;
    onChange: (value: string) => void;
    placeholder?: string;
  }) => (
    <input
      aria-label={placeholder}
      placeholder={placeholder}
      value={value}
      onChange={(event) => onChange(event.target.value)}
    />
  ),
}));

vi.mock("@/components/ui/table", () => ({
  Table: ({ children }: React.PropsWithChildren) => <table>{children}</table>,
  TableBody: ({ children }: React.PropsWithChildren) => <tbody>{children}</tbody>,
  TableCell: ({ children, ...props }: React.PropsWithChildren<React.TdHTMLAttributes<HTMLTableCellElement>>) => (
    <td {...props}>{children}</td>
  ),
  TableHead: ({ children, ...props }: React.PropsWithChildren<React.ThHTMLAttributes<HTMLTableCellElement>>) => (
    <th {...props}>{children}</th>
  ),
  TableHeader: ({ children }: React.PropsWithChildren) => <thead>{children}</thead>,
  TableRow: ({ children }: React.PropsWithChildren) => <tr>{children}</tr>,
}));

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({
    open,
    children,
  }: React.PropsWithChildren<{ open: boolean; onOpenChange: (open: boolean) => void }>) =>
    open ? <div>{children}</div> : null,
  DialogContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogDescription: ({ children }: React.PropsWithChildren) => <p>{children}</p>,
  DialogFooter: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogTitle: ({ children }: React.PropsWithChildren) => <h2>{children}</h2>,
}));

vi.mock("@/components/ui/tabs", async () => {
  const ReactModule = await import("react");
  const TabsContext = ReactModule.createContext<{
    value: string;
    onValueChange?: (value: string) => void;
  }>({ value: "all" });

  return {
    Tabs: ({
      children,
      value,
      onValueChange,
    }: React.PropsWithChildren<{ value: string; onValueChange?: (value: string) => void }>) => (
      <TabsContext.Provider value={{ value, onValueChange }}>
        <div>{children}</div>
      </TabsContext.Provider>
    ),
    TabsList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
    TabsTrigger: ({
      children,
      value,
    }: React.PropsWithChildren<{ value: string }>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return (
        <button type="button" aria-pressed={ctx.value === value} onClick={() => ctx.onValueChange?.(value)}>
          {children}
        </button>
      );
    },
  };
});

vi.mock("@/components/ui/tooltip", () => ({
  TooltipProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
  Tooltip: ({ children }: React.PropsWithChildren) => <>{children}</>,
  TooltipTrigger: ({ children }: React.PropsWithChildren) => <>{children}</>,
  TooltipContent: ({ children }: React.PropsWithChildren) => <>{children}</>,
}));

vi.mock("@/components/authorization/HintIcon", () => ({
  HintIcon: ({ children }: React.PropsWithChildren<{ label: string }>) => <span>{children}</span>,
}));

vi.mock("@/components/ui/tooltip-tag-list", () => ({
  TooltipTagList: () => <div>tag list</div>,
}));

vi.mock("@/components/authorization/ApproveWithContextDialog", () => ({
  ApproveWithContextDialog: ({
    agent,
    onApprove,
  }: {
    agent: AgentTagSummary | null;
    policies: AccessPolicy[];
    onApprove: (agentId: string, selectedTags: string[]) => Promise<void>;
    onOpenChange: (open: boolean) => void;
  }) =>
    agent ? (
      <div>
        <div>Approve dialog for {agent.agent_id}</div>
        <button type="button" onClick={() => void onApprove(agent.agent_id, ["approved-tag"])}>
          Confirm approve
        </button>
      </div>
    ) : null,
}));

vi.mock("@/components/authorization/RevokeDialog", () => ({
  RevokeDialog: ({
    agent,
    onRevoke,
  }: {
    agent: AgentTagSummary | null;
    onRevoke: (agentId: string, reason?: string) => Promise<void>;
    onOpenChange: (open: boolean) => void;
  }) =>
    agent ? (
      <div>
        <div>Revoke dialog for {agent.agent_id}</div>
        <button type="button" onClick={() => void onRevoke(agent.agent_id, "cleanup")}>
          Confirm revoke
        </button>
      </div>
    ) : null,
}));

function makeAgent(overrides: Partial<AgentTagSummary>): AgentTagSummary {
  return {
    agent_id: "agent-1",
    proposed_tags: ["alpha"],
    approved_tags: [],
    lifecycle_status: "pending_approval",
    registered_at: "2026-04-01T00:00:00Z",
    ...overrides,
  };
}

describe("AgentTagsTab", () => {
  beforeEach(() => {
    tagsState.approveAgentTags.mockReset();
    tagsState.rejectAgentTags.mockReset();
    tagsState.revokeAgentTags.mockReset();
    tagsState.success.mockReset();
    tagsState.error.mockReset();
    tagsState.approveAgentTags.mockResolvedValue({});
    tagsState.rejectAgentTags.mockResolvedValue({});
    tagsState.revokeAgentTags.mockResolvedValue({});
  });

  it("renders errors and read-only state", () => {
    const { rerender } = render(
      <AgentTagsTab
        policies={[]}
        agents={[]}
        agentsLoading={false}
        agentsError={new Error("agents failed")}
        canMutate
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText("agents failed")).toBeInTheDocument();

    rerender(
      <AgentTagsTab
        policies={[]}
        agents={[makeAgent({ lifecycle_status: "active", approved_tags: ["ops"] })]}
        agentsLoading={false}
        agentsError={null}
        canMutate={false}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText("View only")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Revoke" })).not.toBeInTheDocument();
  });

  it("filters agents and handles approve, reject, and revoke flows", async () => {
    const onRefresh = vi.fn();
    const agents = [
      makeAgent({ agent_id: "agent-pending", proposed_tags: ["needs-review"] }),
      makeAgent({
        agent_id: "agent-live",
        proposed_tags: ["ops"],
        approved_tags: ["ops"],
        lifecycle_status: "active",
      }),
    ];

    render(
      <AgentTagsTab
        policies={[]}
        agents={agents}
        agentsLoading={false}
        agentsError={null}
        canMutate
        onRefresh={onRefresh}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText("Search agent id or tags…"), {
      target: { value: "pending" },
    });
    expect(screen.getByText("agent-pending")).toBeInTheDocument();
    expect(screen.queryByText("agent-live")).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Search agent id or tags…"), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByRole("button", { name: /pending/i }));
    fireEvent.click(screen.getByRole("button", { name: "Approve" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm approve" }));

    await waitFor(() => {
      expect(tagsState.approveAgentTags).toHaveBeenCalledWith("agent-pending", {
        approved_tags: ["approved-tag"],
      });
    });

    fireEvent.click(screen.getByRole("button", { name: "Reject" }));
    fireEvent.change(screen.getByPlaceholderText("Reason for rejection"), {
      target: { value: "policy mismatch" },
    });
    fireEvent.click(screen.getAllByRole("button", { name: "Reject" })[1]);

    await waitFor(() => {
      expect(tagsState.rejectAgentTags).toHaveBeenCalledWith("agent-pending", {
        reason: "policy mismatch",
      });
    });

    fireEvent.click(screen.getByRole("button", { name: /approved/i }));
    fireEvent.click(screen.getByRole("button", { name: "Revoke" }));
    fireEvent.click(screen.getByRole("button", { name: "Confirm revoke" }));

    await waitFor(() => {
      expect(tagsState.revokeAgentTags).toHaveBeenCalledWith("agent-live", "cleanup");
    });

    expect(tagsState.success).toHaveBeenCalledWith("Tags approved for agent-pending");
    expect(tagsState.success).toHaveBeenCalledWith("Tags rejected for agent-pending");
    expect(tagsState.success).toHaveBeenCalledWith("Tags revoked for agent-live");
    expect(onRefresh).toHaveBeenCalledTimes(3);
  });
});
