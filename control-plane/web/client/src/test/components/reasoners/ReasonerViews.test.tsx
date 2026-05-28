// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExecutionHistoryList } from "@/components/reasoners/ExecutionHistoryList";
import { JsonModal } from "@/components/reasoners/JsonModal";
import { ReasonerGrid } from "@/components/reasoners/ReasonerGrid";
import { SmartStringRenderer } from "@/components/reasoners/SmartStringRenderer";

const navigateMock = vi.fn();
const clipboardWriteTextMock = vi.fn();
const createObjectUrlMock = vi.fn(() => "blob:mock");
const revokeObjectUrlMock = vi.fn();

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock,
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  CheckmarkFilled: () => <span>check</span>,
  ErrorFilled: () => <span>error</span>,
  Time: () => <span>time</span>,
  Copy: () => <span>copy</span>,
  WarningFilled: () => <span>warning</span>,
  Launch: () => <span>launch</span>,
  InProgress: () => <span>progress</span>,
  PauseFilled: () => <span>pause</span>,
  ChevronDown: () => <span>chevron-down</span>,
  ChevronRight: () => <span>chevron-right</span>,
  Code: () => <span>code</span>,
  Document: () => <span>document</span>,
  Maximize: () => <span>maximize</span>,
  View: () => <span>view</span>,
  Download: () => <span>download</span>,
  Close: () => <span>close</span>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button type="button" onClick={onClick} {...props}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/badge", () => ({
  Badge: ({ children }: React.PropsWithChildren) => <span>{children}</span>,
}));

vi.mock("@/components/ui/table", () => ({
  Table: ({ children }: React.PropsWithChildren) => <table>{children}</table>,
  TableHeader: ({ children }: React.PropsWithChildren) => <thead>{children}</thead>,
  TableBody: ({ children }: React.PropsWithChildren) => <tbody>{children}</tbody>,
  TableRow: ({
    children,
    onClick,
    onKeyDown,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLTableRowElement>>) => (
    <tr onClick={onClick} onKeyDown={onKeyDown} {...props}>
      {children}
    </tr>
  ),
  TableHead: ({ children }: React.PropsWithChildren) => <th>{children}</th>,
  TableCell: ({ children }: React.PropsWithChildren) => <td>{children}</td>,
}));

vi.mock("@/utils/status", () => ({
  normalizeExecutionStatus: (status: string) => {
    if (status === "completed") return "succeeded";
    return status ?? "unknown";
  },
  getStatusLabel: (status: string) => `label:${status}`,
}));

vi.mock("@/components/layout/ResponsiveGrid", () => ({
  ResponsiveGrid: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: () => <div>skeleton</div>,
}));

vi.mock("@/components/did/DIDDisplay", () => ({
  DIDIdentityBadge: ({ nodeId }: { nodeId: string }) => <span>did:{nodeId}</span>,
}));

vi.mock("@/components/reasoners/EmptyReasonersState", () => ({
  EmptyReasonersState: ({ type }: { type: string }) => <div>empty:{type}</div>,
}));

vi.mock("@/components/reasoners/ReasonerCard", () => ({
  ReasonerCard: ({
    reasoner,
    onClick,
  }: {
    reasoner: { name: string };
    onClick?: (reasoner: unknown) => void;
  }) => (
    <button type="button" onClick={() => onClick?.(reasoner)}>
      card:{reasoner.name}
    </button>
  ),
}));

vi.mock("@/components/reasoners/ReasonerStatusDot", () => ({
  ReasonerStatusDot: ({ status }: { status: string }) => <span>status:{status}</span>,
}));

vi.mock("@/components/ui/collapsible", () => ({
  Collapsible: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleTrigger: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/segmented-control", () => ({
  SegmentedControl: ({
    value,
    onValueChange,
    options,
  }: {
    value: string;
    onValueChange: (value: string) => void;
    options: ReadonlyArray<{ value: string; label: string }>;
  }) => (
    <div>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onValueChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  ),
}));

vi.mock("@/components/ui/json-syntax-highlight", () => ({
  JsonHighlightedPre: ({ text, data }: { text?: string; data?: unknown }) => (
    <pre>{text ?? JSON.stringify(data, null, 2)}</pre>
  ),
}));

describe("reasoner-related views", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    if (!navigator.clipboard) {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: {},
      });
    }
    Object.defineProperty(navigator.clipboard, "writeText", {
      configurable: true,
      writable: true,
      value: clipboardWriteTextMock,
    });
    clipboardWriteTextMock.mockResolvedValue(undefined);
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: createObjectUrlMock,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: revokeObjectUrlMock,
    });
  });

  it("renders history states and supports row actions", async () => {
    const loadMore = vi.fn();

    const { rerender } = render(<ExecutionHistoryList history={null} />);
    expect(screen.getByText(/Loading execution history/i)).toBeInTheDocument();

    rerender(
      <ExecutionHistoryList history={{ executions: [], total: 0, page: 1, limit: 10 }} />
    );
    expect(screen.getByText(/No executions found/i)).toBeInTheDocument();

    rerender(
      <ExecutionHistoryList
        history={{
          executions: [
            {
              execution_id: "execution-123456789",
              duration_ms: 2500,
              status: "completed",
              timestamp: new Date().toISOString(),
              cost: 0.125,
              error_message: "boom",
              input: {},
            },
          ],
          total: 2,
          page: 1,
          limit: 1,
        }}
        onLoadMore={loadMore}
      />
    );

    fireEvent.click(screen.getAllByRole("button", { name: /Copy execution ID/i })[0]);
    expect(screen.getAllByText("label:succeeded").length).toBeGreaterThan(0);
    expect(clipboardWriteTextMock).toHaveBeenCalledWith("execution-123456789");

    await userEvent
      .setup()
      .click(screen.getAllByRole("button", { name: /View execution details/i })[0]);
    expect(navigateMock).toHaveBeenCalledWith("/executions/execution-123456789");

    await userEvent.setup().click(screen.getByText(/Load More Executions/i));
    expect(loadMore).toHaveBeenCalledTimes(1);
    expect(screen.getAllByText("label:succeeded")[0]).toBeInTheDocument();
    expect(screen.getByText("Error:")).toBeInTheDocument();
  });

  it("renders grid, table, and empty states for reasoners", async () => {
    const onReasonerClick = vi.fn();
    const reasoner = {
      reasoner_id: "node.planner",
      name: "Planner",
      description: "Plans",
      node_id: "node-1",
      node_status: "active",
      avg_response_time_ms: 12,
      success_rate: 0.98,
      last_updated: new Date().toISOString(),
    };

    const { rerender } = render(<ReasonerGrid reasoners={[]} />);
    expect(screen.getByText("empty:no-reasoners")).toBeInTheDocument();

    rerender(<ReasonerGrid reasoners={[reasoner] as never} onReasonerClick={onReasonerClick} />);
    await userEvent.setup().click(screen.getByRole("button", { name: /card:Planner/i }));
    expect(onReasonerClick).toHaveBeenCalledWith(reasoner);

    rerender(
      <ReasonerGrid
        reasoners={[reasoner] as never}
        onReasonerClick={onReasonerClick}
        viewMode="table"
      />
    );
    await userEvent.setup().click(screen.getByText("Open →"));
    expect(onReasonerClick).toHaveBeenCalledWith(reasoner);
    expect(screen.getByText("status:online")).toBeInTheDocument();
    expect(screen.getByText("did:node-1")).toBeInTheDocument();
  });

  it("renders markdown and url smart strings and opens modal callbacks", async () => {
    const onOpenModal = vi.fn();
    const user = userEvent.setup();

    const { rerender } = render(
      <SmartStringRenderer
        content="https://example.com/docs"
        label="url"
        path={["result"]}
        onOpenModal={onOpenModal}
      />
    );

    expect(screen.getByRole("link", { name: "https://example.com/docs" })).toHaveAttribute(
      "href",
      "https://example.com/docs"
    );
    fireEvent.click(screen.getByRole("button"));
    expect(clipboardWriteTextMock).toHaveBeenCalledWith("https://example.com/docs");

    rerender(
      <SmartStringRenderer
        content={"# Heading\n\nconst value = 1;\n".repeat(8)}
        label="markdown"
        path={["result"]}
        onOpenModal={onOpenModal}
      />
    );

    expect(screen.getByText(/chars/i)).toBeInTheDocument();
    expect(screen.getByText("Markdown")).toBeInTheDocument();
    expect(screen.getByText("Code")).toBeInTheDocument();

    const buttons = screen.getAllByRole("button");
    expect(screen.getAllByText("Heading").length).toBeGreaterThan(0);
    await user.click(buttons[2]);
    expect(onOpenModal).toHaveBeenCalledTimes(1);
  });

  it("renders JsonModal formatted and raw views, copies content, and downloads", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    const appendSpy = vi.spyOn(document.body, "appendChild");
    const removeSpy = vi.spyOn(document.body, "removeChild");
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(
      <JsonModal
        isOpen={true}
        onClose={onClose}
        content={"# Title\n\nPlain text"}
        path={["root", "message"]}
        title="Execution output"
      />
    );

    expect(screen.getByText("Execution output")).toBeInTheDocument();
    expect(screen.getByText("Title")).toBeInTheDocument();
    expect(screen.getByText("root.message")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Copy/i }));
    expect(clipboardWriteTextMock).toHaveBeenCalledWith("# Title\n\nPlain text");

    await user.click(screen.getByRole("button", { name: /Download/i }));
    expect(createObjectUrlMock).toHaveBeenCalled();
    expect(appendSpy).toHaveBeenCalled();
    expect(clickSpy).toHaveBeenCalled();
    expect(removeSpy).toHaveBeenCalled();
    expect(revokeObjectUrlMock).toHaveBeenCalledWith("blob:mock");

    await user.click(screen.getByRole("button", { name: /Raw/i }));
    expect(
      screen.getAllByText((_, element) => element?.textContent === "# Title\n\nPlain text").length
    ).toBeGreaterThan(0);

    fireEvent.click(screen.getByText("close").closest("button") as HTMLButtonElement);
    expect(onClose).toHaveBeenCalledTimes(1);

    clickSpy.mockRestore();
  });
});