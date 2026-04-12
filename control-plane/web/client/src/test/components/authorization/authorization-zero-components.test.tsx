// @ts-nocheck
import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PolicyContextPanel } from "@/components/authorization/PolicyContextPanel";
import { RevokeDialog } from "@/components/authorization/RevokeDialog";

vi.mock("@/components/ui/badge", () => ({
  Badge: ({
    children,
    ...props
  }: React.PropsWithChildren<React.HTMLAttributes<HTMLSpanElement>>) => <span {...props}>{children}</span>,
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

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({
    children,
    open,
  }: React.PropsWithChildren<{ open?: boolean; onOpenChange?: (open: boolean) => void }>) =>
    open ? <div role="dialog">{children}</div> : null,
  DialogContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  DialogTitle: ({ children }: React.PropsWithChildren) => <h2>{children}</h2>,
  DialogDescription: ({ children }: React.PropsWithChildren) => <p>{children}</p>,
  DialogFooter: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/input", () => ({
  Input: React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
    (props, ref) => <input ref={ref} {...props} />
  ),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, ...props }: React.PropsWithChildren<React.LabelHTMLAttributes<HTMLLabelElement>>) => (
    <label {...props}>{children}</label>
  ),
}));

vi.mock("@/lib/utils", () => ({
  cn: (...values: Array<string | false | null | undefined>) => values.filter(Boolean).join(" "),
}));

vi.mock("@/lib/theme", () => ({
  statusTone: {
    success: { border: "success-border" },
    error: { border: "error-border" },
  },
}));

describe("authorization zero-coverage components", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders policy context empty states and matched policies", () => {
    const policies = [
      {
        id: "p-1",
        name: "Allow alpha to beta",
        action: "allow",
        caller_tags: ["alpha"],
        target_tags: ["beta"],
        allow_functions: ["read"],
        deny_functions: [],
      },
      {
        id: "p-2",
        name: "Deny beta to gamma",
        action: "deny",
        caller_tags: ["beta"],
        target_tags: ["gamma"],
        allow_functions: [],
        deny_functions: ["write"],
      },
    ];

    const { rerender } = render(<PolicyContextPanel tags={[]} policies={policies as never} />);
    expect(screen.getByText("Select at least one tag to see policy impact.")).toBeInTheDocument();

    rerender(<PolicyContextPanel tags={["delta"]} policies={policies as never} />);
    expect(screen.getByText("No existing policies reference these tags.")).toBeInTheDocument();

    rerender(<PolicyContextPanel tags={["alpha", "gamma"]} policies={policies as never} />);
    expect(screen.getByText("As Caller")).toBeInTheDocument();
    expect(screen.getByText("As Target")).toBeInTheDocument();
    expect(screen.getByText("Allow alpha to beta")).toBeInTheDocument();
    expect(screen.getByText("Deny beta to gamma")).toBeInTheDocument();
    expect(screen.getByText(/Functions: read/)).toBeInTheDocument();
    expect(screen.getByText(/Functions: !write/)).toBeInTheDocument();
  });

  it("renders revoke dialog and revokes tags with a reason", async () => {
    const user = userEvent.setup();
    const onRevoke = vi.fn().mockResolvedValue(undefined);
    const onOpenChange = vi.fn();

    render(
      <RevokeDialog
        agent={{
          agent_id: "agent-1",
          approved_tags: ["prod", "writer"],
        } as never}
        onRevoke={onRevoke}
        onOpenChange={onOpenChange}
      />
    );

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Current Approved Tags")).toBeInTheDocument();
    expect(screen.getByText("prod")).toBeInTheDocument();

    await user.type(screen.getByLabelText("Reason (optional)"), "manual review");
    await user.click(screen.getByRole("button", { name: "Revoke Tags" }));

    expect(onRevoke).toHaveBeenCalledWith("agent-1", "manual review");
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("closes revoke dialog when cancel is clicked", async () => {
    const user = userEvent.setup();
    const onOpenChange = vi.fn();

    render(<RevokeDialog agent={null} onRevoke={vi.fn()} onOpenChange={onOpenChange} />);
    expect(screen.queryByRole("dialog")).toBeNull();

    render(
      <RevokeDialog
        agent={{ agent_id: "agent-2", approved_tags: [] } as never}
        onRevoke={vi.fn().mockResolvedValue(undefined)}
        onOpenChange={onOpenChange}
      />
    );

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
});
