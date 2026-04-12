// @ts-nocheck
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ApproveWithContextDialog } from "@/components/authorization/ApproveWithContextDialog";
import { DataModal, EnhancedModal } from "@/components/execution/EnhancedModal";
import { EmptyReasonersState } from "@/components/reasoners/EmptyReasonersState";
import { ErrorState } from "@/components/ui/ErrorState";
import { WorkflowDeleteDialog } from "@/components/workflows/WorkflowDeleteDialog";

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/lib/theme", () => ({
  statusTone: {
    error: {
      accent: "error-accent",
      bg: "error-bg",
      border: "error-border",
      fg: "error-fg",
    },
    warning: {
      accent: "warning-accent",
      bg: "warning-bg",
      border: "warning-border",
      fg: "warning-fg",
    },
    info: {
      accent: "info-accent",
      bg: "info-bg",
      border: "info-border",
      fg: "info-fg",
    },
  },
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status?: string) => status ?? "unknown",
}));

vi.mock("@/components/ui/icon-bridge", async () => {
  const ReactModule = await import("react");
  const Icon = ReactModule.forwardRef<SVGSVGElement, { className?: string }>(function Icon(props, ref) {
    return <svg ref={ref} data-testid="icon" {...props} />;
  });

  return {
    Trash: Icon,
    WarningOctagon: Icon,
    SpinnerGap: Icon,
    ArrowsOutSimple: Icon,
    CornersIn: Icon,
    X: Icon,
    CheckCircle: Icon,
    Wifi: Icon,
    WifiOff: Icon,
    Grid: Icon,
    Terminal: Icon,
    Renew: Icon,
    Search: Icon,
    CloudOffline: Icon,
    AlertTriangle: Icon,
    RefreshCw: Icon,
  };
});

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    type = "button",
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type={type} onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({
    children,
    variant,
    showIcon,
    ...props
  }: React.PropsWithChildren<{ variant?: string; showIcon?: boolean } & React.HTMLAttributes<HTMLSpanElement>>) => (
    <span data-variant={variant} {...props}>
      {children}
    </span>
  ),
}));

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => (
    <h3 {...props}>{children}</h3>
  ),
  CardContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }: React.PropsWithChildren<React.LabelHTMLAttributes<HTMLLabelElement>>) => (
    <label {...props}>{children}</label>
  ),
}));

vi.mock("@/components/authorization/PolicyContextPanel", () => ({
  PolicyContextPanel: ({ tags }: { tags: string[] }) => <div>{`policy:${tags.join(",")}`}</div>,
}));

vi.mock("@/components/ui/UnifiedJsonViewer", () => ({
  UnifiedJsonViewer: ({ data }: { data: unknown }) => <div>{`formatted:${JSON.stringify(data)}`}</div>,
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ text }: { text: string }) => <pre>{text}</pre>,
}));

vi.mock("@/components/ui/empty", () => ({
  Empty: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  EmptyHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  EmptyMedia: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  EmptyTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => (
    <h3 {...props}>{children}</h3>
  ),
  EmptyDescription: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLParagraphElement>>) => <p {...props}>{children}</p>,
  EmptyContent: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
}));

vi.mock("@/components/ui/dialog", async () => {
  const ReactModule = await import("react");
  const DialogContext = ReactModule.createContext<{ open: boolean; onOpenChange?: (open: boolean) => void }>({
    open: false,
  });

  return {
    Dialog: ({
      children,
      open = true,
      onOpenChange,
    }: React.PropsWithChildren<{ open?: boolean; onOpenChange?: (open: boolean) => void }>) => (
      <DialogContext.Provider value={{ open, onOpenChange }}>{children}</DialogContext.Provider>
    ),
    DialogContent: ({
      children,
      ...props
    }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => {
      const ctx = ReactModule.useContext(DialogContext);
      if (!ctx.open) {
        return null;
      }
      return (
        <div role="dialog" {...props}>
          {children}
          <button type="button" aria-label="Close" onClick={() => ctx.onOpenChange?.(false)}>
            Close
          </button>
        </div>
      );
    },
    DialogHeader: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
      <div {...props}>{children}</div>
    ),
    DialogFooter: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
      <div {...props}>{children}</div>
    ),
    DialogTitle: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLHeadingElement>>) => (
      <h2 {...props}>{children}</h2>
    ),
    DialogDescription: ({
      children,
      ...props
    }: React.PropsWithChildren<React.HTMLAttributes<HTMLParagraphElement>>) => <p {...props}>{children}</p>,
  };
});

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
      defaultValue,
      ...props
    }: React.PropsWithChildren<{
      value?: string;
      onValueChange?: (value: string) => void;
      defaultValue?: string;
    }>) => {
      const [internalValue, setInternalValue] = ReactModule.useState(defaultValue ?? value ?? "");
      const currentValue = value ?? internalValue;
      const handleChange = onValueChange ?? setInternalValue;
      return (
        <TabsContext.Provider value={{ value: currentValue, onValueChange: handleChange }}>
          <div {...props}>{children}</div>
        </TabsContext.Provider>
      );
    },
    TabsList: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
      <div {...props}>{children}</div>
    ),
    TabsTrigger: ({
      children,
      value,
      ...props
    }: React.PropsWithChildren<{ value: string } & React.ButtonHTMLAttributes<HTMLButtonElement>>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return (
        <button type="button" role="tab" onClick={() => ctx.onValueChange?.(value)} {...props}>
          {children}
        </button>
      );
    },
    TabsContent: ({
      children,
      value,
      ...props
    }: React.PropsWithChildren<{ value: string } & React.HTMLAttributes<HTMLDivElement>>) => {
      const ctx = ReactModule.useContext(TabsContext);
      return ctx.value === value ? <div {...props}>{children}</div> : null;
    },
  };
});

describe("ui modals and empty states", () => {
  it("handles workflow delete cancel, confirm, and id filtering", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const onConfirm = vi.fn().mockResolvedValue([]);

    render(
      <WorkflowDeleteDialog
        isOpen={true}
        onClose={onClose}
        onConfirm={onConfirm}
        workflows={[
          {
            workflow_id: "wf-1",
            run_id: "run-fallback",
            display_name: "Primary Workflow",
            agent_name: "agent-a",
            total_executions: 3,
            status: "running",
            status_counts: { running: 1, succeeded: 2 },
          },
          {
            workflow_id: undefined,
            run_id: "run-2",
            display_name: "Fallback Workflow",
            agent_name: "agent-b",
            total_executions: 2,
            status: "failed",
            status_counts: { failed: 1, timeout: 1 },
          },
        ]}
      />,
    );

    expect(screen.getByText("Delete workflows")).toBeInTheDocument();
    expect(screen.getByText(/2 workflows to delete/i)).toBeInTheDocument();
    expect(screen.getByText(/1 in-flight execution will be force-cancelled/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /Delete 2 workflows/i }));
    expect(onConfirm).toHaveBeenCalledWith(["wf-1", "run-2"]);
    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(2));
  });

  it("keeps workflow delete dialog open during deletion and shows errors on failure", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    let rejectDelete: ((reason?: unknown) => void) | undefined;
    const onConfirm = vi.fn(
      () =>
        new Promise<never>((_, reject) => {
          rejectDelete = reject;
        }),
    );

    render(
      <WorkflowDeleteDialog
        isOpen={true}
        onClose={onClose}
        onConfirm={onConfirm}
        workflows={[
          {
            workflow_id: "wf-1",
            run_id: "run-1",
            display_name: "Workflow",
            agent_name: "agent-a",
            total_executions: 1,
            status: "running",
            status_counts: { running: 1 },
          },
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Delete workflow/i }));
    expect(screen.getByText("Deleting...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).not.toHaveBeenCalled();

    rejectDelete?.(new Error("delete failed"));
    await waitFor(() => expect(screen.getByText("delete failed")).toBeInTheDocument());
    expect(onClose).not.toHaveBeenCalled();
  });

  it("toggles the enhanced modal maximize state and closes from the header button", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(
      <EnhancedModal isOpen={true} onClose={onClose} title="Execution Detail">
        <div>modal body</div>
      </EnhancedModal>,
    );

    expect(screen.getByText("modal body")).toBeInTheDocument();
    expect(screen.getByTitle("Maximize")).toBeInTheDocument();

    await user.click(screen.getByTitle("Maximize"));
    expect(screen.getByTitle("Restore")).toBeInTheDocument();

    await user.click(screen.getByTitle("Close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("renders data modal tabs and markdown preview content", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const markdown = "**Bold**\n\n`code`\n\n[Docs](https://example.com)";

    render(<DataModal isOpen={true} onClose={onClose} title="Payload" data={markdown} />);

    expect(screen.getByText("Payload - Full View")).toBeInTheDocument();
    expect(screen.getByText(`formatted:${JSON.stringify(markdown)}`)).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Raw JSON" }));
    expect(screen.getByText(JSON.stringify(markdown, null, 2))).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: "Markdown Preview" }));
    expect(screen.getByRole("link", { name: "Docs" })).toHaveAttribute("href", "https://example.com");
    expect(screen.getByText("Bold").tagName).toBe("STRONG");

    await user.click(screen.getByTitle("Close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("approves selected tags and supports cancel in the context dialog", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn().mockResolvedValue(undefined);
    const onOpenChange = vi.fn();

    render(
      <ApproveWithContextDialog
        agent={{
          agent_id: "agent-1",
          proposed_tags: ["finance", "ops"],
        }}
        policies={[]}
        onApprove={onApprove}
        onOpenChange={onOpenChange}
      />,
    );

    expect(screen.getByText("Approve Tags")).toBeInTheDocument();
    expect(screen.getByText("policy:finance,ops")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /ops/i }));
    expect(screen.getByText("policy:finance")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);

    await user.click(screen.getByRole("button", { name: /Approve 1 tag/i }));
    expect(onApprove).toHaveBeenCalledWith("agent-1", ["finance"]);
    await waitFor(() => expect(onOpenChange).toHaveBeenCalledWith(false));
  });

  it("approves an agent with no requested tags", async () => {
    const user = userEvent.setup();
    const onApprove = vi.fn().mockResolvedValue(undefined);
    const onOpenChange = vi.fn();

    render(
      <ApproveWithContextDialog
        agent={{
          agent_id: "agent-2",
          proposed_tags: [],
        }}
        policies={[]}
        onApprove={onApprove}
        onOpenChange={onOpenChange}
      />,
    );

    expect(screen.getByRole("heading", { name: "Approve Agent" })).toBeInTheDocument();
    expect(screen.queryByText(/Policy Impact/i)).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Approve Agent" }));
    expect(onApprove).toHaveBeenCalledWith("agent-2", []);
  });

  it("renders empty reasoner states and fires the available actions", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    const onShowAll = vi.fn();
    const onClearFilters = vi.fn();

    const { rerender } = render(
      <EmptyReasonersState type="no-online" onRefresh={onRefresh} onShowAll={onShowAll} />,
    );

    expect(screen.getByText("No Online Reasoners")).toBeInTheDocument();
    expect(screen.getByText("Connection check")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show All Reasoners" }));
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(onShowAll).toHaveBeenCalledTimes(1);
    expect(onRefresh).toHaveBeenCalledTimes(1);

    rerender(
      <EmptyReasonersState
        type="no-search-results"
        searchTerm="planner"
        onClearFilters={onClearFilters}
        onRefresh={onRefresh}
      />,
    );

    expect(screen.getByText(/No reasoners match "planner"/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Clear Filters" }));
    expect(onClearFilters).toHaveBeenCalledTimes(1);
  });

  it("renders error state variants and fires retry and dismiss callbacks", async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const onDismiss = vi.fn();

    const { rerender } = render(
      <ErrorState
        title="Load failed"
        error={new Error("network down")}
        onRetry={onRetry}
        onDismiss={onDismiss}
      />,
    );

    expect(screen.getByText("Load failed")).toBeInTheDocument();
    expect(screen.getByText("network down")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    await user.click(screen.getByRole("button", { name: "Dismiss" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onDismiss).toHaveBeenCalledTimes(1);

    rerender(
      <ErrorState
        title="Inline warning"
        description="cached fallback"
        variant="inline"
        severity="warning"
        onRetry={onRetry}
        retrying={true}
      />,
    );

    expect(screen.getByText("Inline warning")).toBeInTheDocument();
    expect(screen.getByText("cached fallback")).toBeInTheDocument();
    expect(screen.getByRole("button")).toBeDisabled();

    rerender(
      <ErrorState
        title="Banner info"
        description="refresh available"
        variant="banner"
        severity="info"
        onRetry={onRetry}
        onDismiss={onDismiss}
      />,
    );

    expect(screen.getByText("Banner info")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(onRetry).toHaveBeenCalledTimes(2);
  });
});
