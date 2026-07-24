import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PolicyFormDialog } from "@/components/authorization/PolicyFormDialog";
import type { AccessPolicy } from "@/services/accessPoliciesApi";

const policyApiState = vi.hoisted(() => ({
  listKnownTags: vi.fn(),
}));

vi.mock("@/services/accessPoliciesApi", async () => {
  const actual = await vi.importActual<typeof import("@/services/accessPoliciesApi")>(
    "@/services/accessPoliciesApi",
  );
  return {
    ...actual,
    listKnownTags: (...args: unknown[]) => policyApiState.listKnownTags(...args),
  };
});

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

vi.mock("@/components/ui/chip-input", () => {
  function ChipInput({
    value,
    onChange,
    placeholder,
    suggestions,
  }: {
    value: string[];
    onChange: (value: string[]) => void;
    placeholder?: string;
    suggestions?: string[];
  }) {
    const [draft, setDraft] = React.useState("");
    return (
      <div>
        <input
          aria-label={placeholder}
          placeholder={placeholder}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
        />
        <button
          type="button"
          onClick={() => {
            if (draft.trim()) {
              onChange([...value, draft.trim()]);
              setDraft("");
            }
          }}
        >
          Add {placeholder}
        </button>
        <div data-testid={`chips-${placeholder}`}>{value.join(",")}</div>
        {suggestions?.length ? <div data-testid={`suggestions-${placeholder}`}>{suggestions.join(",")}</div> : null}
      </div>
    );
  }
  return { ChipInput };
});

vi.mock("@/components/ui/icon-bridge", () => {
  const Icon = () => <span>icon</span>;
  return {
    CheckCircle: Icon,
    XCircle: Icon,
    CaretRight: Icon,
  };
});

describe("PolicyFormDialog", () => {
  beforeEach(() => {
    policyApiState.listKnownTags.mockReset();
    policyApiState.listKnownTags.mockResolvedValue({ tags: ["analytics", "finance"], total: 2 });
  });

  it("creates a deny policy and closes the dialog after save", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    const onOpenChange = vi.fn();

    render(
      <PolicyFormDialog open onOpenChange={onOpenChange} editPolicy={null} onSave={onSave} />,
    );

    expect(await screen.findByTestId("suggestions-e.g. analytics")).toHaveTextContent(
      "analytics,finance",
    );

    fireEvent.click(screen.getByRole("button", { name: "Deny" }));
    fireEvent.change(screen.getByPlaceholderText("e.g. analytics-read-financial"), {
      target: { value: "deny-finance" },
    });
    fireEvent.change(screen.getByPlaceholderText("Optional description"), {
      target: { value: "Block finance calls" },
    });
    fireEvent.change(screen.getByPlaceholderText("0"), {
      target: { value: "7" },
    });

    fireEvent.change(screen.getByPlaceholderText("e.g. analytics"), {
      target: { value: "analytics" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add e.g. analytics" }));

    fireEvent.change(screen.getByPlaceholderText("e.g. financial"), {
      target: { value: "finance" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add e.g. financial" }));

    fireEvent.change(screen.getByPlaceholderText("e.g. get_report, list_*"), {
      target: { value: "delete_*" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add e.g. get_report, list_*" }));

    fireEvent.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "deny-finance",
          description: "Block finance calls",
          action: "deny",
          priority: 7,
          caller_tags: ["analytics"],
          target_tags: ["finance"],
          allow_functions: [],
          deny_functions: ["delete_*"],
        }),
        undefined,
      );
    });
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("prefills edit data and submits updates with the policy id", async () => {
    const editPolicy: AccessPolicy = {
      id: 4,
      name: "allow-analytics",
      caller_tags: ["analytics"],
      target_tags: ["reports"],
      allow_functions: ["get_report"],
      deny_functions: [],
      action: "allow",
      priority: 3,
      enabled: true,
      description: "Existing rule",
      created_at: "2026-04-01T00:00:00Z",
      updated_at: "2026-04-01T00:00:00Z",
    };
    const onSave = vi.fn().mockResolvedValue(undefined);

    render(
      <PolicyFormDialog open onOpenChange={vi.fn()} editPolicy={editPolicy} onSave={onSave} />,
    );

    expect(await screen.findByDisplayValue("allow-analytics")).toBeInTheDocument();
    expect(screen.getByDisplayValue("Existing rule")).toBeInTheDocument();
    expect(screen.getByDisplayValue("3")).toBeInTheDocument();
    expect(screen.getByTestId("chips-e.g. analytics")).toHaveTextContent("analytics");
    expect(screen.getByTestId("chips-e.g. financial")).toHaveTextContent("reports");
    expect(screen.getByTestId("chips-e.g. get_report, list_*")).toHaveTextContent("get_report");

    fireEvent.change(screen.getByPlaceholderText("Optional description"), {
      target: { value: "Updated rule" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Update" }));

    await waitFor(() => {
      expect(onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "allow-analytics",
          description: "Updated rule",
          allow_functions: ["get_report"],
        }),
        4,
      );
    });
  });
});
