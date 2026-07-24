// @ts-nocheck
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";

import { PageHeader } from "@/components/PageHeader";
import { CommandPalette } from "@/components/CommandPalette";
import FloatingConnectionLine from "@/components/WorkflowDAG/FloatingConnectionLine";
import { ReasonerStatusDot } from "@/components/reasoners/ReasonerStatusDot";

const navigate = vi.fn();
const getBezierPath = vi.fn();
const getEdgeParams = vi.fn();

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigate,
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

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

vi.mock("@/components/ui/FilterSelect", () => ({
  FilterSelect: ({
    label,
    value,
    options,
    onValueChange,
  }: {
    label: string;
    value: string;
    options: Array<{ label: string; value: string }>;
    onValueChange: (value: string) => void;
  }) => (
    <label>
      {label}
      <select aria-label={label} value={value} onChange={(e) => onValueChange(e.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  ),
}));

vi.mock("@/components/ui/command", () => ({
  CommandDialog: ({
    children,
    open,
  }: React.PropsWithChildren<{ open?: boolean; onOpenChange?: (open: boolean) => void }>) =>
    open ? <div role="dialog">{children}</div> : null,
  CommandEmpty: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CommandGroup: ({
    children,
    heading,
  }: React.PropsWithChildren<{ heading: string }>) => (
    <section>
      <h2>{heading}</h2>
      {children}
    </section>
  ),
  CommandInput: (props: React.InputHTMLAttributes<HTMLInputElement>) => (
    <input aria-label="command-input" {...props} />
  ),
  CommandItem: ({
    children,
    onSelect,
  }: React.PropsWithChildren<{ onSelect?: () => void }>) => (
    <button type="button" onClick={onSelect}>
      {children}
    </button>
  ),
  CommandList: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CommandSeparator: () => <hr />,
}));

vi.mock("lucide-react", () => {
  const Icon = React.forwardRef<SVGSVGElement, React.SVGProps<SVGSVGElement>>((props, ref) => (
    <svg ref={ref} {...props} />
  ));
  return {
    LayoutDashboard: Icon,
    Play: Icon,
    Server: Icon,
    FlaskConical: Icon,
    KeyRound: Icon,
    FileCheck2: Icon,
    Settings: Icon,
    Search: Icon,
  };
});

vi.mock("@xyflow/react", () => ({
  getBezierPath: (...args: unknown[]) => getBezierPath(...args),
}));

vi.mock("@/components/WorkflowDAG/EdgeUtils", () => ({
  getEdgeParams: (...args: unknown[]) => getEdgeParams(...args),
}));

describe("misc ui zero-coverage components", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getEdgeParams.mockReturnValue({
      sx: 10,
      sy: 20,
      tx: 110,
      ty: 120,
      sourcePos: "right",
      targetPos: "left",
    });
    getBezierPath.mockReturnValue(["M10,20 C40,50 70,80 110,120"]);
  });

  it("renders page header with actions, filters, aside, and view options", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    const onFilterChange = vi.fn();

    render(
      <PageHeader
        title="Runs"
        description="Inspect workflow runs"
        aside={<span>beta</span>}
        actions={[{ label: "Refresh", onClick: onRefresh }]}
        filters={[
          {
            label: "Status",
            value: "all",
            options: [{ label: "All", value: "all" }, { label: "Running", value: "running" }],
            onChange: onFilterChange,
          },
        ]}
        viewOptions={<button type="button">grid</button>}
      />
    );

    expect(screen.getByText("Runs")).toBeInTheDocument();
    expect(screen.getByText("Inspect workflow runs")).toBeInTheDocument();
    expect(screen.getByText("beta")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "grid" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(onRefresh).toHaveBeenCalledTimes(1);

    await user.selectOptions(screen.getByLabelText("Status"), "running");
    expect(onFilterChange).toHaveBeenCalledWith("running");
  });

  it("opens command palette with ctrl+k and navigates from actions", async () => {
    const user = userEvent.setup();
    render(<CommandPalette />);

    fireEvent.keyDown(document, { ctrlKey: true, key: "k" });
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Navigation")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /show failed runs/i }));
    expect(navigate).toHaveBeenCalledWith("/runs?status=failed");
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).toBeNull();
    });
  });

  it("renders the floating connection line path and cursor target", () => {
    const { container, rerender } = render(
      <svg>
        <FloatingConnectionLine
          toX={200}
          toY={220}
          fromPosition="right"
          toPosition="left"
          fromNode={{ id: "node-1" }}
        />
      </svg>
    );

    expect(getEdgeParams).toHaveBeenCalled();
    expect(container.querySelector("path")).toHaveAttribute("d", "M10,20 C40,50 70,80 110,120");
    expect(container.querySelector("circle")).toHaveAttribute("cx", "110");

    rerender(
      <svg>
        <FloatingConnectionLine
          toX={200}
          toY={220}
          fromPosition="right"
          toPosition="left"
          fromNode={null}
        />
      </svg>
    );
    expect(container.querySelector("path")).toBeNull();
  });

  it("renders reasoner status dots for text and dot-only modes", () => {
    const { rerender, container } = render(<ReasonerStatusDot status="online" />);
    expect(screen.getByText("Online")).toBeInTheDocument();

    rerender(<ReasonerStatusDot status="degraded" size="lg" />);
    expect(screen.getByText("Limited")).toBeInTheDocument();

    rerender(<ReasonerStatusDot status="offline" showText={false} size="sm" />);
    expect(screen.queryByText("Offline")).toBeNull();
    expect(container.querySelector(".bg-muted")).toBeInTheDocument();

    rerender(<ReasonerStatusDot status="unknown" />);
    expect(screen.getByText("Unknown")).toBeInTheDocument();
  });
});
