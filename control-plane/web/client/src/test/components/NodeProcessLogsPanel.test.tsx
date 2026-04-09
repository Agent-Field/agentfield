import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { NodeProcessLogsPanel } from "@/components/nodes/NodeProcessLogsPanel";
import type { NodeLogEntry } from "@/services/api";

const state = vi.hoisted(() => {
  // NodeProcessLogsPanel does `e instanceof NodeLogsError` in its error
  // branch. Without exporting a NodeLogsError constructor from the mock,
  // the reference is undefined at runtime and `instanceof` throws a
  // TypeError — which swallows the destructive-alert render and fails
  // unrelated assertions in this suite. Provide a minimal shape-compatible
  // mock class.
  class MockNodeLogsError extends Error {
    readonly status: number;
    readonly code?: string;
    constructor(message: string, status: number, code?: string) {
      super(message);
      this.name = "NodeLogsError";
      this.status = status;
      this.code = code;
    }
  }
  return {
    fetchNodeLogsText: vi.fn(),
    parseNodeLogsNDJSON: vi.fn(),
    streamNodeLogsEntries: vi.fn(),
    NodeLogsError: MockNodeLogsError,
  };
});

vi.mock("@/services/api", () => ({
  fetchNodeLogsText: (...args: unknown[]) => state.fetchNodeLogsText(...args),
  parseNodeLogsNDJSON: (...args: unknown[]) => state.parseNodeLogsNDJSON(...args),
  streamNodeLogsEntries: (...args: unknown[]) => state.streamNodeLogsEntries(...args),
  NodeLogsError: state.NodeLogsError,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  AlertDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
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
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <h2>{children}</h2>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/collapsible", () => ({
  Collapsible: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CollapsibleTrigger: ({ children }: React.PropsWithChildren) => <button type="button">{children}</button>,
  CollapsibleContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/dropdown-menu", () => ({
  DropdownMenu: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuTrigger: ({ children }: React.PropsWithChildren) => <>{children}</>,
  DropdownMenuContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DropdownMenuItem: ({
    children,
    onClick,
    disabled,
  }: React.PropsWithChildren<{ onClick?: () => void; disabled?: boolean }>) => (
    <button type="button" disabled={disabled} onClick={onClick}>
      {children}
    </button>
  ),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({
    children,
    ...props
  }: React.PropsWithChildren<React.LabelHTMLAttributes<HTMLLabelElement>>) => <label {...props}>{children}</label>,
}));

vi.mock("@/components/ui/scroll-area", () => {
  const ReactModule = require("react") as typeof React;
  return {
    ScrollArea: ReactModule.forwardRef<
      HTMLDivElement,
      React.PropsWithChildren<{ className?: string }>
    >(({ children, className }, ref) => (
      <div ref={ref} className={className}>
        <div data-radix-scroll-area-viewport>{children}</div>
      </div>
    )),
  };
});

vi.mock("@/components/ui/segmented-control", () => ({
  SegmentedControl: ({
    value,
    onValueChange,
    options,
  }: {
    value: string;
    onValueChange: (value: string) => void;
    options: Array<{ value: string; label: string }>;
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

vi.mock("@/components/ui/SearchBar", () => ({
  SearchBar: ({
    value,
    onChange,
    inputClassName,
    ...props
  }: {
    value: string;
    onChange: (value: string) => void;
    inputClassName?: string;
  } & React.InputHTMLAttributes<HTMLInputElement>) => (
    <input
      value={value}
      onChange={(event) => onChange(event.target.value)}
      {...props}
    />
  ),
}));

vi.mock("@/components/authorization/HintIcon", () => ({
  HintIcon: ({ children }: React.PropsWithChildren<{ label: string }>) => <span>{children}</span>,
}));

vi.mock("@/components/execution/observabilityStyles", () => ({
  observabilityStyles: new Proxy(
    {},
    {
      get: () => "",
    },
  ),
}));

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = ({ className }: { className?: string }) => <svg data-testid="icon" className={className} />;
  return {
    AlertCircle: Icon,
    ChevronDown: Icon,
    ChevronRight: Icon,
    Copy: Icon,
    Download: Icon,
    MoreHorizontal: Icon,
    PauseCircle: Icon,
    Play: Icon,
    RefreshCw: Icon,
    Terminal: Icon,
  };
});

beforeAll(() => {
  Object.assign(navigator, {
    clipboard: {
      writeText: vi.fn(),
    },
  });
  Object.assign(URL, {
    createObjectURL: vi.fn(),
    revokeObjectURL: vi.fn(),
  });
});

describe("NodeProcessLogsPanel", () => {
  beforeEach(() => {
    state.fetchNodeLogsText.mockReset();
    state.parseNodeLogsNDJSON.mockReset();
    state.streamNodeLogsEntries.mockReset();
    vi.restoreAllMocks();
  });

  it("loads logs, filters them, copies and downloads visible rows, and appends live entries", async () => {
    const user = userEvent.setup();
    const clipboardWrite = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined);
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const createObjectURL = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:logs");
    const revokeObjectURL = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    const entries: NodeLogEntry[] = [
      {
        v: 1,
        seq: 1,
        ts: "2026-04-08T10:00:00.000Z",
        stream: "stdout",
        line: JSON.stringify({
          v: 1,
          execution_id: "exec-1",
          run_id: "run-1",
          event_type: "started",
          message: "Structured hello",
          source: "sdk",
          level: "info",
        }),
      },
      {
        v: 1,
        seq: 2,
        ts: "2026-04-08T10:00:01.000Z",
        stream: "stderr",
        line: "plain failure",
        level: "error",
        source: "worker",
      },
    ];

    state.fetchNodeLogsText.mockResolvedValue("ignored");
    state.parseNodeLogsNDJSON.mockReturnValue(entries);
    state.streamNodeLogsEntries.mockImplementation(
      async function* () {
        yield {
          v: 1,
          seq: 3,
          ts: "2026-04-08T10:00:02.000Z",
          stream: "stdout",
          line: JSON.stringify({
            v: 1,
            execution_id: "exec-2",
            event_type: "live_update",
            message: "streamed line",
            source: "streamer",
            level: "info",
          }),
          source: "streamer",
        };
      },
    );

    render(<NodeProcessLogsPanel nodeId="node-1" />);

    expect(state.fetchNodeLogsText).toHaveBeenCalledWith("node-1", { tail_lines: "200" });

    expect(await screen.findByText("Structured hello")).toBeInTheDocument();
    expect(screen.getByText("plain failure")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Structured (1)" })[0]);
    expect(screen.getByText("Structured hello")).toBeInTheDocument();
    expect(screen.queryByText("plain failure")).not.toBeInTheDocument();

    await user.type(screen.getByRole("textbox", { name: "Filter log lines by text" }), "exec-1");
    expect(screen.getByDisplayValue("exec-1")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Copy" })[0]);
    expect(clipboardWrite).toHaveBeenCalledWith(expect.stringContaining("\"seq\":1"));

    await user.click(screen.getAllByRole("button", { name: "Download" })[0]);
    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(anchorClick).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:logs");

    await user.clear(screen.getByRole("textbox", { name: "Filter log lines by text" }));
    await user.click(screen.getAllByRole("button", { name: "All (2)" })[0]);
    await user.click(screen.getAllByRole("button", { name: "Live" })[0]);

    expect(await screen.findByText("streamed line")).toBeInTheDocument();
  });

  it("shows a destructive alert when the tail request fails", async () => {
    state.fetchNodeLogsText.mockRejectedValue(new Error("tail failed"));
    state.parseNodeLogsNDJSON.mockReturnValue([]);

    render(<NodeProcessLogsPanel nodeId="node-err" />);

    await waitFor(() => {
      expect(screen.getByText("Logs unavailable")).toBeInTheDocument();
    });
    expect(screen.getByText("tail failed")).toBeInTheDocument();
  });
});
