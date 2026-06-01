// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExecutionForm } from "@/components/reasoners/ExecutionForm";

const mocks = vi.hoisted(() => {
  type WatchCallback = (values: Record<string, unknown>) => void;
  let watchCallback: WatchCallback | null = null;
  let currentValues: Record<string, unknown> = {};

  return {
    jsonSchemaToZodObjectMock: vi.fn(),
    ZodProviderMock: vi.fn(function MockProvider(this: Record<string, unknown>, schema: unknown) {
      this.schema = schema;
    }),
    resetMock: vi.fn((values: Record<string, unknown>) => {
      currentValues = values;
    }),
    getValuesMock: vi.fn(() => currentValues),
    setWatchCallback: (callback: WatchCallback | null) => {
      watchCallback = callback;
    },
    emitWatch: (values: Record<string, unknown>) => {
      currentValues = values;
      watchCallback?.(values);
    },
    resetState: () => {
      watchCallback = null;
      currentValues = {};
    },
  };
});

function emitWatch(values: Record<string, unknown>) {
  mocks.emitWatch(values);
}

vi.mock("@/utils/jsonSchemaToZod", () => ({
  jsonSchemaToZodObject: (...args: unknown[]) => mocks.jsonSchemaToZodObjectMock(...args),
}));

vi.mock("@autoform/zod", () => ({
  ZodProvider: mocks.ZodProviderMock,
}));

vi.mock("@autoform/react", () => ({
  AutoForm: ({ onFormInit }: { onFormInit: (form: unknown) => void }) => {
    onFormInit({
      reset: mocks.resetMock,
      getValues: mocks.getValuesMock,
      watch: (callback: WatchCallback) => {
        mocks.setWatchCallback(callback);
        return { unsubscribe: vi.fn() };
      },
    });
    return <div>mock-auto-form</div>;
  },
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    ...props
  }: React.PropsWithChildren<React.ButtonHTMLAttributes<HTMLButtonElement>>) => (
    <button {...props}>{children}</button>
  ),
}));

vi.mock("@/components/ui/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => <input {...props} />,
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, htmlFor }: React.PropsWithChildren<{ htmlFor?: string }>) => (
    <label htmlFor={htmlFor}>{children}</label>
  ),
}));

vi.mock("@/components/ui/select", () => ({
  Select: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  SelectContent: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  SelectItem: ({ children }: React.PropsWithChildren) => <div>{children}</div>,
  SelectTrigger: ({ children, id }: React.PropsWithChildren<{ id?: string }>) => <div id={id}>{children}</div>,
  SelectValue: ({ placeholder }: { placeholder?: string }) => <span>{placeholder}</span>,
}));

vi.mock("@/components/ui/switch", () => ({
  Switch: ({ checked, onCheckedChange }: { checked?: boolean; onCheckedChange?: (value: boolean) => void }) => (
    <button type="button" aria-label="switch" onClick={() => onCheckedChange?.(!checked)}>
      {checked ? "on" : "off"}
    </button>
  ),
}));

vi.mock("@/components/ui/auto-expanding-textarea", () => ({
  AutoExpandingTextarea: React.forwardRef<
    HTMLTextAreaElement,
    React.TextareaHTMLAttributes<HTMLTextAreaElement>
  >((props, ref) => <textarea ref={ref} {...props} />),
}));

vi.mock("@/components/ui/icon-bridge", () => ({
  WarningFilled: () => <span>warning</span>,
}));

describe("ExecutionForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.resetState();
  });

  it("renders fallback raw JSON editor when no schema provider is available", () => {
    const onChange = vi.fn();

    render(
      <ExecutionForm
        formData={{ input: { foo: "bar" } }}
        onChange={onChange}
        validationErrors={["first issue", "second issue"]}
      />
    );

    expect(screen.getByText("Validation errors")).toBeInTheDocument();
    expect(screen.getByText(/Structured form rendering is unavailable/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Raw Input JSON")).toHaveValue('{\n  "foo": "bar"\n}');

    fireEvent.change(screen.getByLabelText("Raw Input JSON"), {
      target: { value: "" },
    });
    expect(screen.getByText("JSON cannot be empty")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Raw Input JSON"), {
      target: { value: "{invalid" },
    });
    expect(screen.getByText("Invalid JSON")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Raw Input JSON"), {
      target: { value: '{"foo":"baz"}' },
    });

    expect(onChange).toHaveBeenCalled();
    const update = onChange.mock.calls.at(-1)?.[0] as (previous: Record<string, unknown>) => Record<string, unknown>;
    expect(update({ existing: true })).toEqual({ existing: true, input: { foo: "baz" } });
  });

  it("attaches the structured form watcher and syncs watched values back to onChange", () => {
    const onChange = vi.fn();

    mocks.jsonSchemaToZodObjectMock.mockReturnValue({ type: "object" });
    mocks.resetMock.mockImplementation((values: Record<string, unknown>) => {
      mocks.getValuesMock.mockReturnValue(values);
    });
    mocks.getValuesMock.mockReturnValue({ foo: "bar" });

    const { rerender } = render(
      <ExecutionForm
        schema={{ type: "object", properties: { foo: { type: "string" } } }}
        formData={{ input: { foo: "bar" } }}
        onChange={onChange}
        validationErrors={[]}
      />
    );

    expect(screen.getByText("mock-auto-form")).toBeInTheDocument();
    expect(mocks.resetMock).toHaveBeenCalledWith({ foo: "bar" });

    emitWatch({ foo: "baz" });

    const update = onChange.mock.calls.at(-1)?.[0] as (previous: Record<string, unknown>) => Record<string, unknown>;
    expect(update({ input: { foo: "bar" }, extra: 1 })).toEqual({
      input: { foo: "baz" },
      extra: 1,
    });

    rerender(
      <ExecutionForm
        schema={{ type: "object", properties: { foo: { type: "string" } } }}
        formData={{ input: { foo: "server" } }}
        onChange={onChange}
        validationErrors={[]}
      />
    );

    expect(mocks.resetMock).toHaveBeenLastCalledWith({ foo: "server" });
  });
});