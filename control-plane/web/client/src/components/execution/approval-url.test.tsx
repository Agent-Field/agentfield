import type { PropsWithChildren } from "react";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";
import { afterEach, expect, it, vi } from "vitest";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { WorkflowExecution } from "@/types/executions";
import { CompactExecutionHeader } from "./CompactExecutionHeader";
import { ExecutionApprovalPanel } from "./ExecutionApprovalPanel";

vi.mock("@/components/ui/notification", () => ({
  useErrorNotification: () => vi.fn(),
  useSuccessNotification: () => vi.fn(),
}));

afterEach(cleanup);

function Wrapper({ children }: PropsWithChildren) {
  return <MemoryRouter><TooltipProvider>{children}</TooltipProvider></MemoryRouter>;
}

function execution(url: string): WorkflowExecution {
  return {
    id: 1, execution_id: "execution", workflow_id: "workflow", agentfield_request_id: "request",
    agent_node_id: "agent", reasoner_id: "reasoner", workflow_depth: 0,
    input_data: null, output_data: null, input_size: 0, output_size: 0, workflow_tags: [],
    status: "waiting", started_at: "2026-10-01T00:00:00Z", retry_count: 0,
    created_at: "2026-10-01T00:00:00Z", updated_at: "2026-10-01T00:00:00Z",
    approval_request_id: "approval", approval_request_url: url,
  };
}

for (const component of ["panel", "header"] as const) {
  it.each(["javascript:alert(1)", "java\tscript:alert(1)", "data:text/html,payload", "https://user:pass@example.com"])(
    `C2: approval ${component} rejects agent-controlled %s`, async (url) => {
      const { container } = render(component === "panel" ? <ExecutionApprovalPanel execution={execution(url)} /> :
        <CompactExecutionHeader execution={execution(url)} activeTab="approval" onTabChange={() => {}} navigationTabs={[]} />, { wrapper: Wrapper });
      if (component === "panel") await userEvent.click(screen.getByText("Request Details"));
      expect(screen.getByText(component === "panel" ? "Open in Hub" : "Approval Required")).toBeInTheDocument();
      expect(container.querySelector("a[href]")).toBeNull();
    },
  );

  it(`C2: approval ${component} retains safe https review links`, async () => {
    const url = "https://example.com/review";
    render(component === "panel" ? <ExecutionApprovalPanel execution={execution(url)} /> :
      <CompactExecutionHeader execution={execution(url)} activeTab="approval" onTabChange={() => {}} navigationTabs={[]} />, { wrapper: Wrapper });
    if (component === "panel") await userEvent.click(screen.getByText("Request Details"));
    const link = screen.getByRole("link", { name: component === "panel" ? "Open in Hub" : "Approval Required" });
    expect(link).toHaveAttribute("href", url);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });
}
