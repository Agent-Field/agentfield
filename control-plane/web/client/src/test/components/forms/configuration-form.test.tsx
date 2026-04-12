import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ConfigField } from "@/components/forms/ConfigField";
import { ConfigurationForm } from "@/components/forms/ConfigurationForm";
import type { ConfigurationSchema } from "@/types/agentfield";

vi.mock("@/components/ui/icon-bridge", () => ({
  Eye: () => <span>eye</span>,
  EyeOff: () => <span>eye-off</span>,
  Loader2: () => <span>loader</span>,
  Save: () => <span>save</span>,
  AlertCircle: () => <span>alert</span>,
}));

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

vi.mock("@/components/ui/card", () => ({
  Card: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
    <div {...props}>{children}</div>
  ),
  CardHeader: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardTitle: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  CardContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children }: React.PropsWithChildren) => <div role="alert">{children}</div>,
  AlertDescription: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/input", () => ({
  Input: React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
    ({ ...props }, ref) => <input ref={ref} {...props} />
  ),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({
    children,
    htmlFor,
    ...props
  }: React.PropsWithChildren<{ htmlFor?: string } & React.LabelHTMLAttributes<HTMLLabelElement>>) => (
    <label htmlFor={htmlFor} {...props}>
      {children}
    </label>
  ),
}));

vi.mock("@/components/ui/switch", () => ({
  Switch: ({
    checked,
    onCheckedChange,
  }: {
    checked: boolean;
    onCheckedChange: (value: boolean) => void;
  }) => (
    <button type="button" aria-pressed={checked} onClick={() => onCheckedChange(!checked)}>
      {checked ? "on" : "off"}
    </button>
  ),
}));

vi.mock("@/components/ui/select", () => {
  const React = require("react") as typeof import("react");
  const SelectContext = React.createContext<{
    value: string;
    onValueChange: (value: string) => void;
  } | null>(null);

  return {
    Select: ({
      value,
      onValueChange,
      children,
    }: React.PropsWithChildren<{ value: string; onValueChange: (value: string) => void }>) => (
      <SelectContext.Provider value={{ value, onValueChange }}>{children}</SelectContext.Provider>
    ),
    SelectTrigger: ({ children, ...props }: React.PropsWithChildren<React.HTMLAttributes<HTMLDivElement>>) => (
      <div {...props}>{children}</div>
    ),
    SelectValue: ({ placeholder }: { placeholder?: string }) => <span>{placeholder}</span>,
    SelectContent: ({ children }: React.PropsWithChildren) => {
      const ctx = React.useContext(SelectContext);
      return (
        <select
          aria-label="select-field"
          value={ctx?.value ?? ""}
          onChange={(event) => ctx?.onValueChange(event.target.value)}
        >
          <option value="">Select</option>
          {children}
        </select>
      );
    },
    SelectItem: ({
      children,
      value,
    }: React.PropsWithChildren<{ value: string }>) => <option value={value}>{children}</option>,
  };
});

describe("configuration form and fields", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders config field variants and propagates changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    const { rerender } = render(
      <ConfigField
        field={{ name: "apiKey", type: "secret", description: "Key", required: true }}
        value="secret-value"
        onChange={onChange}
        error="Required"
      />
    );

    const secretInput = screen.getByDisplayValue("secret-value");
    expect(secretInput).toHaveAttribute("type", "password");
    expect(screen.getByText("Required")).toBeInTheDocument();

    await user.click(screen.getByRole("button"));
    expect(secretInput).toHaveAttribute("type", "text");

    rerender(
      <ConfigField
        field={{ name: "enabled", type: "boolean" }}
        value={false}
        onChange={onChange}
      />
    );
    await user.click(screen.getByRole("button", { name: "off" }));
    expect(onChange).toHaveBeenCalledWith(true);

    rerender(
      <ConfigField
        field={{
          name: "mode",
          type: "select",
          options: [
            { value: "safe", label: "Safe" },
            { value: "fast", label: "Fast" },
          ],
        }}
        value="safe"
        onChange={onChange}
      />
    );
    await user.selectOptions(screen.getByLabelText("select-field"), "fast");
    expect(onChange).toHaveBeenCalledWith("fast");

    rerender(
      <ConfigField
        field={{ name: "limit", type: "number", validation: { min: 1, max: 10 } }}
        value={2}
        onChange={onChange}
      />
    );
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: "7" } });
    expect(onChange).toHaveBeenCalledWith(7);
  });

  it("initializes defaults, validates input, submits values, and surfaces submit errors", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    const schema: ConfigurationSchema = {
      fields: [
        { name: "username", type: "text", required: true, validation: { pattern: "^[a-z]+$" } },
        { name: "retries", type: "number", required: true, default: 3, validation: { min: 1, max: 5 } },
        { name: "enabled", type: "boolean", default: true },
      ],
    };

    render(
      <ConfigurationForm
        schema={schema}
        initialValues={{ username: "alice" }}
        onSubmit={onSubmit}
        title="Agent Config"
        description="Configure it"
      />
    );

    expect(screen.getByText("Agent Config")).toBeInTheDocument();
    expect(screen.getByText("Configure it")).toBeInTheDocument();
    expect(screen.getByDisplayValue("alice")).toBeInTheDocument();
    expect(screen.getByDisplayValue("3")).toBeInTheDocument();

    await user.clear(screen.getByDisplayValue("alice"));
    await user.type(screen.getByRole("textbox"), "Alice-1");
    await user.clear(screen.getByDisplayValue("3"));
    await user.type(screen.getByRole("spinbutton"), "0");
    await user.click(screen.getByRole("button", { name: /Save Configuration/i }));

    expect(screen.getByText("username format is invalid")).toBeInTheDocument();
    expect(screen.getByText("retries must be at least 1")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();

    await user.clear(screen.getByRole("textbox"));
    await user.type(screen.getByRole("textbox"), "alice");
    await user.clear(screen.getByRole("spinbutton"));
    await user.type(screen.getByRole("spinbutton"), "5");
    await user.click(screen.getByRole("button", { name: /Save Configuration/i }));

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith({
        username: "alice",
        retries: 5,
        enabled: true,
      });
    });
  });

  it("shows submit error messages from failed form submission", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn().mockRejectedValue(new Error("save failed"));

    render(
      <ConfigurationForm
        schema={{ fields: [{ name: "username", type: "text", required: true }] }}
        initialValues={{ username: "alice" }}
        onSubmit={onSubmit}
      />
    );

    await user.click(screen.getByRole("button", { name: /Save Configuration/i }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent("save failed");
    });
  });
});
