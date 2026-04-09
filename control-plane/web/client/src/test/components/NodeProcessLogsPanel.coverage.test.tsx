// @ts-nocheck
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import { NodeProcessLogsPanel } from "@/components/nodes/NodeProcessLogsPanel";

const state = vi.hoisted(() => {
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
});

describe("NodeProcessLogsPanel coverage paths", () => {
  beforeEach(() => {
    state.fetchNodeLogsText.mockReset();
    state.parseNodeLogsNDJSON.mockReset();
    state.streamNodeLogsEntries.mockReset();
  });

  it("shows other-stream guidance and warns when structured logs are hidden by the stream filter", async () => {
    const user = userEvent.setup();
    state.fetchNodeLogsText.mockResolvedValue("ignored");
    state.parseNodeLogsNDJSON.mockReturnValue([
      {
        v: 1,
        seq: 1,
        ts: "2026-04-08T10:00:00.000Z",
        stream: "stdout",
        line: JSON.stringify({
          execution_id: "exec-1",
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
        stream: "custom",
        line: "other stream line",
      },
    ]);

    render(<NodeProcessLogsPanel nodeId="node-2" />);

    expect(await screen.findByText("Structured hello")).toBeInTheDocument();
    expect(
      screen.getByText((content, node) => {
        const text = node?.textContent ?? content;
        return text === "1 line on other streams, shown only in All.";
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Structured (1)" }));
    expect(screen.getByText("Structured hello")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Stderr" }));
    expect(
      screen.getByText("Structured logs are available on a different stream"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The current stream filter is hiding them/i),
    ).toBeInTheDocument();
  });

  it("renders load-tail and live-stream errors, then recovers on refresh", async () => {
    const user = userEvent.setup();
    state.fetchNodeLogsText
      .mockRejectedValueOnce(new Error("tail failed"))
      .mockResolvedValueOnce("ignored");
    state.parseNodeLogsNDJSON.mockReturnValue([]);
    state.streamNodeLogsEntries.mockImplementation(async function* () {
      throw new Error("stream interrupted");
    });

    render(<NodeProcessLogsPanel nodeId="node-3" />);

    expect(await screen.findByText("Logs unavailable")).toBeInTheDocument();
    expect(screen.getByText("tail failed")).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Refresh" })[0]);
    await waitFor(() => {
      expect(state.fetchNodeLogsText).toHaveBeenCalledTimes(2);
    });
    expect(screen.getByText(/No log lines yet/i)).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Live" })[0]);
    expect(await screen.findByText("stream interrupted")).toBeInTheDocument();
  });

  it("treats 404 NodeLogsError as empty state, not an error (TC-035)", async () => {
    state.fetchNodeLogsText.mockRejectedValueOnce(
      new state.NodeLogsError("HTTP 404", 404),
    );
    state.parseNodeLogsNDJSON.mockReturnValue([]);

    render(<NodeProcessLogsPanel nodeId="node-404" />);

    // Friendly empty state, not the destructive "Logs unavailable" alert.
    expect(await screen.findByText(/No log lines yet/i)).toBeInTheDocument();
    expect(screen.queryByText("Logs unavailable")).not.toBeInTheDocument();
  });

  it("treats agent_unreachable (no base_url) as empty state (TC-035)", async () => {
    state.fetchNodeLogsText.mockRejectedValueOnce(
      new state.NodeLogsError("node has no base_url", 502, "agent_unreachable"),
    );
    state.parseNodeLogsNDJSON.mockReturnValue([]);

    render(<NodeProcessLogsPanel nodeId="node-no-baseurl" />);

    expect(await screen.findByText(/No log lines yet/i)).toBeInTheDocument();
    expect(screen.queryByText("Logs unavailable")).not.toBeInTheDocument();
    // The raw backend message must not leak into the UI.
    expect(screen.queryByText("node has no base_url")).not.toBeInTheDocument();
  });
});
