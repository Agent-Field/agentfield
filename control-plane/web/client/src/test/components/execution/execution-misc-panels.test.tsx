// @ts-nocheck
import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { EnhancedDataPanel } from "@/components/execution/EnhancedDataPanel";
import { ExecutionStatusBar } from "@/components/execution/ExecutionStatusBar";
import { ExecutionWebhookActivity } from "@/components/execution/ExecutionWebhookActivity";
import { RedesignedErrorPanel } from "@/components/execution/RedesignedErrorPanel";

const navigate = vi.fn();
const dataModalSpy = vi.fn();

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigate,
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = (props: React.HTMLAttributes<HTMLSpanElement>) => <span {...props} />;
  return {
    ArrowDown: Icon,
    ArrowUp: Icon,
    Database: Icon,
    FileText: Icon,
    Eye: Icon,
    Maximize2: Icon,
    RadioTower: Icon,
    AlertTriangle: Icon,
    CheckCircle2: Icon,
    ArrowLeft: Icon,
    Clock: Icon,
    RotateCcw: Icon,
    Share2: Icon,
    CheckCircle: Icon,
    XCircle: Icon,
    Loader2: Icon,
  };
});

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    disabled,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} disabled={disabled} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children, className }: React.PropsWithChildren<{ className?: string }>) => (
    <span data-class={className}>{children}</span>
  ),
}));

vi.mock("@/components/ui/copy-button", () => ({
  CopyButton: ({ value, tooltip }: { value: string; tooltip?: string }) => (
    <button type="button" aria-label={tooltip ?? "copy"}>
      copy:{value}
    </button>
  ),
}));

vi.mock("@/components/execution/CollapsibleSection", () => ({
  CollapsibleSection: ({
    children,
    title,
    badge,
  }: React.PropsWithChildren<{ title: string; badge?: React.ReactNode }>) => (
    <section>
      <h2>{title}</h2>
      {badge}
      {children}
    </section>
  ),
}));

vi.mock("@/components/ui/UnifiedJsonViewer", () => ({
  UnifiedJsonViewer: ({ data }: { data: unknown }) => <pre>{JSON.stringify(data)}</pre>,
}));

vi.mock("@/components/execution/EnhancedModal", () => ({
  DataModal: (props: Record<string, unknown>) => {
    dataModalSpy(props);
    return props.isOpen ? <div>modal-open</div> : null;
  },
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ data }: { data: unknown }) => <pre>{JSON.stringify(data)}</pre>,
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status?: string) => {
    if (status === "completed") return "succeeded";
    return status ?? "unknown";
  },
  getStatusLabel: (status: string) => status.toUpperCase(),
  getStatusTheme: () => ({
    iconClass: "icon",
    badgeVariant: "outline",
    pillClass: "pill",
  }),
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

function buildExecution(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    workflow_id: "wf-1",
    execution_id: "exec-1234567890",
    agentfield_request_id: "req-1",
    session_id: "session-1",
    actor_id: "actor-1",
    agent_node_id: "node-1",
    workflow_depth: 0,
    reasoner_id: "planner",
    input_data: { prompt: "hello" },
    output_data: { result: "ok" },
    input_size: 1536,
    output_size: 2048,
    workflow_tags: [],
    status: "completed",
    started_at: "2026-04-08T00:00:00Z",
    created_at: "2026-04-08T00:00:00Z",
    updated_at: "2026-04-08T00:01:00Z",
    retry_count: 0,
    duration_ms: 65000,
    ...overrides,
  };
}

describe("execution misc panels", () => {
  let clipboardWriteText: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { href: "https://app.test/executions/exec-123" },
    });
    if (!navigator.clipboard) {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: { writeText: async () => undefined },
      });
    }
    clipboardWriteText = vi.spyOn(navigator.clipboard, "writeText").mockResolvedValue(undefined);
  });

  it("renders data preview and opens the modal for large content", async () => {
    const user = userEvent.setup();
    render(
      <EnhancedDataPanel
        execution={buildExecution({
          input_data: { large: "x".repeat(1200) },
          input_size: 2048,
        }) as never}
        type="input"
      />
    );

    expect(screen.getByText("Input Data")).toBeInTheDocument();
    expect(screen.getByText("2.0 KB")).toBeInTheDocument();
    expect(screen.getByText("(Showing preview)")).toBeInTheDocument();
    expect(screen.getByText("Data Preview")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /view full/i }));
    expect(screen.getByText("modal-open")).toBeInTheDocument();
    expect(dataModalSpy).toHaveBeenLastCalledWith(
      expect.objectContaining({ isOpen: true, title: "Input Data" })
    );
  });

  it("renders output empty states for running and failed executions", () => {
    const { rerender } = render(
      <EnhancedDataPanel
        execution={buildExecution({ output_data: {}, output_size: 0, status: "running" }) as never}
        type="output"
      />
    );

    expect(screen.getByText("Execution in progress")).toBeInTheDocument();
    expect(
      screen.getByText("Output data will appear here when the execution completes")
    ).toBeInTheDocument();

    rerender(
      <EnhancedDataPanel
        execution={buildExecution({ output_data: {}, output_size: 0, status: "failed" }) as never}
        type="output"
      />
    );

    expect(screen.getByText("Execution failed")).toBeInTheDocument();
    expect(screen.getByText("No output data was generated due to execution failure")).toBeInTheDocument();
  });

  it("renders webhook counts, sorts events, shows errors, and retries", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();

    render(
      <ExecutionWebhookActivity
        execution={buildExecution({
          webhook_registered: true,
          webhook_events: [
            {
              id: 1,
              execution_id: "exec-1",
              event_type: "webhook",
              status: "failed",
              http_status: 500,
              error_message: "server error",
              created_at: "2026-04-08T10:00:00Z",
            },
            {
              id: 2,
              execution_id: "exec-1",
              event_type: "webhook",
              status: "delivered",
              http_status: 200,
              created_at: "2026-04-08T11:00:00Z",
            },
          ],
        }) as never}
        onRetry={onRetry}
        retryError="retry failed"
      />
    );

    expect(screen.getByText("1 delivered")).toBeInTheDocument();
    expect(screen.getByText("1 failed")).toBeInTheDocument();
    const newerTimestamp = new Date("2026-04-08T11:00:00Z").toLocaleString();
    const olderTimestamp = new Date("2026-04-08T10:00:00Z").toLocaleString();
    const newerNode = screen.getByText(newerTimestamp);
    const olderNode = screen.getByText(olderTimestamp);
    expect(newerNode.compareDocumentPosition(olderNode) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(screen.getByText("HTTP 500")).toBeInTheDocument();
    expect(screen.getByText("server error")).toBeInTheDocument();
    expect(screen.getByText("retry failed")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /retry webhook/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("renders webhook pending and missing-registration empty states", () => {
    const { rerender } = render(
      <ExecutionWebhookActivity
        execution={buildExecution({ webhook_registered: true, webhook_events: [] }) as never}
      />
    );

    expect(screen.getByText("pending")).toBeInTheDocument();
    expect(screen.getByText("Webhook registered – waiting for the first delivery.")).toBeInTheDocument();

    rerender(
      <ExecutionWebhookActivity
        execution={buildExecution({ webhook_registered: false, webhook_events: [] }) as never}
      />
    );

    expect(screen.getByText("No webhook was registered for this execution.")).toBeInTheDocument();
  });

  it("handles back navigation, share, retry visibility, and duration formatting", async () => {
    const user = userEvent.setup();
    render(<ExecutionStatusBar execution={buildExecution({ status: "failed" }) as never} />);

    expect(screen.getByText("1m 5s")).toBeInTheDocument();
    expect(
      screen.getByText((_, element) => element?.tagName === "CODE" && element.textContent === "exec-123...7890")
    ).toBeInTheDocument();
    expect(screen.getByText("FAILED")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /retry/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /back/i }));
    expect(navigate).toHaveBeenCalledWith("/executions");

    fireEvent.click(screen.getByRole("button", { name: /share/i }));
    expect(clipboardWriteText).toHaveBeenCalledWith("https://app.test/executions/exec-123");
  });

  it("renders structured and raw error details", () => {
    const structuredError = JSON.stringify({
      message: "Boom",
      type: "ValidationError",
      code: "E_VALIDATION",
      stack: "stack trace",
      context: { field: "name" },
    });

    const { rerender } = render(
      <RedesignedErrorPanel execution={buildExecution({ error_message: structuredError }) as never} />
    );

    expect(screen.getByText("Execution Failed")).toBeInTheDocument();
    expect(screen.getByText("Boom")).toBeInTheDocument();
    expect(screen.getByText("ValidationError")).toBeInTheDocument();
    expect(screen.getByText("E_VALIDATION")).toBeInTheDocument();
    expect(screen.getByText("stack trace")).toBeInTheDocument();
    expect(screen.getByText('{"field":"name"}')).toBeInTheDocument();

    rerender(
      <RedesignedErrorPanel
        execution={buildExecution({ error_message: "x".repeat(240) }) as never}
      />
    );

    expect(screen.getByText("Full Error Message")).toBeInTheDocument();
    expect(screen.getAllByText("x".repeat(240)).length).toBeGreaterThan(1);
  });

  it("returns null when there is no error message", () => {
    const { container } = render(
      <RedesignedErrorPanel execution={buildExecution({ error_message: undefined }) as never} />
    );
    expect(container).toBeEmptyDOMElement();
  });
});