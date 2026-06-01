import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AccessRulesTab } from "@/components/authorization/AccessRulesTab";
import type { AccessPolicy } from "@/services/accessPoliciesApi";

const rulesState = vi.hoisted(() => ({
  createPolicy: vi.fn(),
  updatePolicy: vi.fn(),
  deletePolicy: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
}));

vi.mock("@/services/accessPoliciesApi", async () => {
  const actual = await vi.importActual<typeof import("@/services/accessPoliciesApi")>(
    "@/services/accessPoliciesApi",
  );
  return {
    ...actual,
    createPolicy: (...args: unknown[]) => rulesState.createPolicy(...args),
    updatePolicy: (...args: unknown[]) => rulesState.updatePolicy(...args),
    deletePolicy: (...args: unknown[]) => rulesState.deletePolicy(...args),
  };
});

vi.mock("@/components/ui/notification", () => ({
  useSuccessNotification: () => rulesState.success,
  useErrorNotification: () => rulesState.error,
  NotificationProvider: ({ children }: React.PropsWithChildren) => <>{children}</>,
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
  TooltipTagList: () => <div>tags</div>,
}));

vi.mock("@/components/authorization/PolicyFormDialog", () => ({
  PolicyFormDialog: ({
    open,
    editPolicy,
    onSave,
  }: {
    open: boolean;
    editPolicy?: AccessPolicy | null;
    onSave: (req: {
      name: string;
      caller_tags: string[];
      target_tags: string[];
      allow_functions?: string[];
      deny_functions?: string[];
      action: "allow" | "deny";
      priority?: number;
      description?: string;
    }, editId?: number) => Promise<void>;
  }) =>
    open ? (
      <div>
        <div>{editPolicy ? `Editing ${editPolicy.name}` : "Creating policy"}</div>
        <button
          type="button"
          onClick={() =>
            void onSave(
              {
                name: editPolicy ? `${editPolicy.name}-updated` : "new-policy",
                caller_tags: ["analytics"],
                target_tags: ["finance"],
                allow_functions: editPolicy ? ["read_finance"] : ["list_*"],
                deny_functions: [],
                action: "allow",
                priority: 9,
                description: "dialog save",
              },
              editPolicy?.id,
            )
          }
        >
          Submit dialog
        </button>
      </div>
    ) : null,
}));

function makePolicy(overrides: Partial<AccessPolicy>): AccessPolicy {
  return {
    id: 1,
    name: "alpha-policy",
    caller_tags: ["alpha"],
    target_tags: ["beta"],
    allow_functions: ["read_*"],
    deny_functions: [],
    action: "allow",
    priority: 5,
    enabled: true,
    description: "default description",
    created_at: "2026-04-01T00:00:00Z",
    updated_at: "2026-04-01T00:00:00Z",
    ...overrides,
  };
}

describe("AccessRulesTab", () => {
  beforeEach(() => {
    rulesState.createPolicy.mockReset();
    rulesState.updatePolicy.mockReset();
    rulesState.deletePolicy.mockReset();
    rulesState.success.mockReset();
    rulesState.error.mockReset();
    rulesState.createPolicy.mockResolvedValue({});
    rulesState.updatePolicy.mockResolvedValue({});
    rulesState.deletePolicy.mockResolvedValue(undefined);
  });

  it("renders read-only and fetch-error states", () => {
    const { rerender } = render(
      <AccessRulesTab
        policies={[]}
        loading={false}
        onRefresh={vi.fn()}
        canMutate={false}
        fetchError={null}
      />,
    );

    expect(screen.getByText("Policies unavailable")).toBeInTheDocument();

    rerender(
      <AccessRulesTab
        policies={[]}
        loading={false}
        onRefresh={vi.fn()}
        canMutate
        fetchError={new Error("policy fetch failed")}
      />,
    );

    expect(screen.getByText("policy fetch failed")).toBeInTheDocument();
  });

  it("filters policies and handles create, edit, and delete actions", async () => {
    const onRefresh = vi.fn();
    const policies = [
      makePolicy({ id: 1, name: "zeta-deny", action: "deny", priority: 1, description: "blocks ops" }),
      makePolicy({ id: 2, name: "alpha-allow", action: "allow", priority: 10, description: "reads finance" }),
    ];

    render(
      <AccessRulesTab
        policies={policies}
        loading={false}
        onRefresh={onRefresh}
        canMutate
        fetchError={null}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText("Search policies, tags, functions…"), {
      target: { value: "finance" },
    });
    expect(screen.getByText("alpha-allow")).toBeInTheDocument();
    expect(screen.queryByText("zeta-deny")).not.toBeInTheDocument();

    fireEvent.change(screen.getByPlaceholderText("Search policies, tags, functions…"), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByRole("button", { name: /deny/i }));
    expect(screen.getByText("zeta-deny")).toBeInTheDocument();
    expect(screen.queryByText("alpha-allow")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^All\b/i }));
    fireEvent.click(screen.getByRole("button", { name: "New policy" }));
    fireEvent.click(screen.getByRole("button", { name: "Submit dialog" }));

    await waitFor(() => {
      expect(rulesState.createPolicy).toHaveBeenCalledWith(
        expect.objectContaining({ name: "new-policy", priority: 9 }),
      );
    });
    expect(rulesState.success).toHaveBeenCalledWith('Policy "new-policy" created');
    expect(onRefresh).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getAllByRole("button", { name: "Edit" })[1]);
    fireEvent.click(screen.getByRole("button", { name: "Submit dialog" }));

    await waitFor(() => {
      expect(rulesState.updatePolicy).toHaveBeenCalledWith(
        1,
        expect.objectContaining({ name: "zeta-deny-updated" }),
      );
    });

    fireEvent.click(screen.getAllByTitle("Delete policy")[1]);
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => {
      expect(rulesState.deletePolicy).toHaveBeenCalledWith(1);
    });
    expect(rulesState.success).toHaveBeenCalledWith("Policy deleted");
    expect(onRefresh).toHaveBeenCalledTimes(3);
  });
});
