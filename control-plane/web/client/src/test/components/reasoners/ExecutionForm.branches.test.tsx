// @ts-nocheck
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ExecutionForm } from "@/components/reasoners/ExecutionForm";

const formState = vi.hoisted(() => {
  type WatchCallback = (values: Record<string, unknown>) => void;

  let currentValues: Record<string, unknown> = {};
  let watchCallback: WatchCallback | null = null;

  const updateValue = (name: string | undefined, value: unknown) => {
    if (!name) {
      return;
    }
    currentValues = {
      ...currentValues,
      [name]: value,
    };
    watchCallback?.(currentValues);
  };

  return {
    currentValues: () => currentValues,
    jsonSchemaToZodObjectMock: vi.fn(),
    ZodProviderMock: vi.fn(function MockProvider(this: Record<string, unknown>, schema: unknown) {
      this.schema = schema;
    }),
    resetMock: vi.fn((values: Record<string, unknown>) => {
      currentValues = values;
    }),
    getValuesMock: vi.fn(() => currentValues),
    unsubscribeMock: vi.fn(),
    emitWatch: (values: Record<string, unknown>) => {
      currentValues = values;
      watchCallback?.(values);
    },
    handleFieldChange: (event: { target?: { name?: string; value?: unknown; checked?: unknown } }) => {
      const name = event.target?.name;
      const value =
        event.target && "value" in event.target ? event.target.value : event.target?.checked;
      updateValue(name, value);
    },
    reset: () => {
      currentValues = {};
      watchCallback = null;
    },
    watch: (callback: WatchCallback) => {
      watchCallback = callback;
      return { unsubscribe: formState.unsubscribeMock };
    },
  };
});

vi.mock("@/utils/jsonSchemaToZod", () => ({
  jsonSchemaToZodObject: (...args: unknown[]) => formState.jsonSchemaToZodObjectMock(...args),
}));

vi.mock("@autoform/zod", () => ({
  ZodProvider: formState.ZodProviderMock,
}));

vi.mock("@/components/ui/alert", () => ({
  Alert: ({ children, ...props }: React.HTMLAttributes<HTMLDivElement>) => <div {...props}>{children}</div>,
}));

vi.mock("@/components/ui/button", () => ({
  Button: React.forwardRef<
    HTMLButtonElement,
    React.ButtonHTMLAttributes<HTMLButtonElement>
  >(({ children, ...props }, ref) => (
    <button ref={ref} {...props}>
      {children}
    </button>
  )),
}));

vi.mock("@/components/ui/input", () => ({
  Input: React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>((props, ref) => (
    <input ref={ref} {...props} />
  )),
}));

vi.mock("@/components/ui/label", () => ({
  Label: ({ children, htmlFor, ...props }: React.LabelHTMLAttributes<HTMLLabelElement>) => (
    <label htmlFor={htmlFor} {...props}>
      {children}
    </label>
  ),
}));

vi.mock("@/components/ui/switch", () => ({
  Switch: React.forwardRef<
    HTMLButtonElement,
    {
      id?: string;
      name?: string;
      checked?: boolean;
      onCheckedChange?: (checked: boolean) => void;
      onBlur?: React.FocusEventHandler<HTMLButtonElement>;
    }
  >(({ id, name, checked, onCheckedChange, onBlur }, ref) => (
    <button
      ref={ref}
      id={id}
      name={name}
      type="button"
      aria-label={name ?? id ?? "switch"}
      onClick={() => onCheckedChange?.(!checked)}
      onBlur={onBlur}
    >
      {checked ? "on" : "off"}
    </button>
  )),
}));

vi.mock("@/components/ui/auto-expanding-textarea", () => ({
  AutoExpandingTextarea: React.forwardRef<
    HTMLTextAreaElement,
    React.TextareaHTMLAttributes<HTMLTextAreaElement>
  >(({ maxHeight: _maxHeight, ...props }, ref) => <textarea ref={ref} {...props} />),
}));

vi.mock("@/components/ui/select", () => {
  const SelectContext = React.createContext<{
    value?: string;
    onValueChange?: (value: string) => void;
  } | null>(null);

  const Select = ({
    value,
    onValueChange,
    children,
  }: React.PropsWithChildren<{ value?: string; onValueChange?: (value: string) => void }>) => {
    return (
      <SelectContext.Provider value={{ value, onValueChange }}>
        <div data-testid="mock-select">{children}</div>
      </SelectContext.Provider>
    );
  };

  const SelectItem = ({ children, value }: React.PropsWithChildren<{ value: string }>) => {
    const context = React.useContext(SelectContext);
    return (
      <button type="button" onClick={() => context?.onValueChange?.(value)}>
        {children}
      </button>
    );
  };

  const SelectTrigger = ({ children, id }: React.PropsWithChildren<{ id?: string }>) => <div id={id}>{children}</div>;
  const SelectContent = ({ children }: React.PropsWithChildren) => <div>{children}</div>;
  const SelectValue = ({ placeholder }: { placeholder?: string }) => {
    const context = React.useContext(SelectContext);
    return <span>{context?.value || placeholder}</span>;
  };

  return { Select, SelectContent, SelectItem, SelectTrigger, SelectValue };
});

vi.mock("@/components/ui/icon-bridge", () => ({
  WarningFilled: React.forwardRef<SVGSVGElement, React.SVGProps<SVGSVGElement>>((props, ref) => (
    <svg ref={ref} aria-label="warning-icon" {...props} />
  )),
}));

vi.mock("@autoform/react", () => {
  const AutoForm = ({
    defaultValues,
    onFormInit,
    formComponents,
    uiComponents,
  }: {
    defaultValues: Record<string, unknown>;
    onFormInit: (form: unknown) => void;
    formComponents: Record<string, React.ComponentType<any>>;
    uiComponents: Record<string, React.ComponentType<any>>;
  }) => {
    React.useEffect(() => {
      formState.resetMock(defaultValues ?? {});
      onFormInit({
        reset: formState.resetMock,
        getValues: formState.getValuesMock,
        watch: formState.watch,
      });
    }, [defaultValues, onFormInit]);

    const StringField = formComponents.string;
    const NumberField = formComponents.number;
    const BooleanField = formComponents.boolean;
    const DateField = formComponents.date;
    const SelectField = formComponents.select;
    const Form = uiComponents.Form;
    const FieldWrapper = uiComponents.FieldWrapper;
    const ErrorMessage = uiComponents.ErrorMessage;
    const SubmitButton = uiComponents.SubmitButton;
    const ObjectWrapper = uiComponents.ObjectWrapper;
    const ArrayWrapper = uiComponents.ArrayWrapper;
    const ArrayElementWrapper = uiComponents.ArrayElementWrapper;

    const baseInputProps = (name: string) => ({
      name,
      onChange: formState.handleFieldChange,
      onBlur: vi.fn(),
    });

    return (
      <Form data-testid="mock-auto-form">
        <FieldWrapper label="Text" id="text" field={{ required: true }}>
          <StringField id="text" value={defaultValues.text} inputProps={baseInputProps("text")} />
        </FieldWrapper>
        <FieldWrapper label="Count" id="count" field={{ required: false }} error="Count issue">
          <NumberField id="count" value={defaultValues.count} inputProps={baseInputProps("count")} />
        </FieldWrapper>
        <FieldWrapper label="Enabled" id="enabled" field={{ required: false }}>
          <BooleanField id="enabled" value={defaultValues.enabled} inputProps={baseInputProps("enabled")} />
        </FieldWrapper>
        <FieldWrapper label="Run Date" id="day" field={{ required: false }}>
          <DateField id="day" value={defaultValues.day} inputProps={baseInputProps("day")} />
        </FieldWrapper>
        <FieldWrapper label="Choice" id="choice" field={{ required: false }}>
          <SelectField
            id="choice"
            value={defaultValues.choice}
            field={{ options: [["beta", "Beta"], ["alpha", "Alpha"]] }}
            inputProps={baseInputProps("choice")}
          />
        </FieldWrapper>
        <FieldWrapper label="Empty Choice" id="emptyChoice" field={{ required: false }}>
          <SelectField
            id="emptyChoice"
            value={defaultValues.emptyChoice}
            field={{ options: [] }}
            inputProps={baseInputProps("emptyChoice")}
          />
        </FieldWrapper>
        <ObjectWrapper label="Advanced group">
          <ArrayWrapper label="Tags" onAddItem={vi.fn()}>
            <ArrayElementWrapper index={0} onRemove={vi.fn()}>
              <div>Array content</div>
            </ArrayElementWrapper>
          </ArrayWrapper>
        </ObjectWrapper>
        <ErrorMessage error="Inline error" />
        <SubmitButton>Apply</SubmitButton>
      </Form>
    );
  };

  return { AutoForm };
});

describe("ExecutionForm branch coverage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    formState.reset();
    formState.jsonSchemaToZodObjectMock.mockReturnValue({ mocked: "zod-object" });
  });

  it("renders structured field components and syncs changes through the watcher", () => {
    const onChange = vi.fn();

    render(
      <ExecutionForm
        schema={{ type: "object", properties: { text: { type: "string" } } }}
        formData={{
          input: {
            text: "seed",
            count: 5,
            enabled: false,
            day: "2026-04-08T12:00:00.000Z",
            choice: "alpha",
          },
        }}
        onChange={onChange}
        validationErrors={["Missing context"]}
      />
    );

    expect(screen.getByTestId("mock-auto-form")).toBeInTheDocument();
    expect(screen.getByText("Validation errors")).toBeInTheDocument();
    expect(screen.getByText(/Missing context/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Text/)).toHaveValue("seed");
    expect(screen.getByLabelText(/Count/)).toHaveValue(5);
    expect(screen.getByLabelText(/Run Date/)).toHaveValue("2026-04-08");
    expect(screen.getByText("Count issue")).toBeInTheDocument();
    expect(screen.getByText("Inline error")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "enabled" })).toHaveTextContent("off");
    expect(screen.getByText("False")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Apply" })).toBeInTheDocument();
    expect(screen.getByText("Add item")).toBeInTheDocument();
    expect(screen.getByText("Remove")).toBeInTheDocument();
    expect(screen.getByText("Value")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Text/), { target: { value: "updated" } });
    let update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update({ input: { text: "seed" }, extra: 1 })).toEqual({
      input: expect.objectContaining({ text: "updated" }),
      extra: 1,
    });

    fireEvent.change(screen.getByLabelText(/Count/), { target: { value: "" } });
    update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update({ input: { count: 5 } }).input.count).toBe("");

    const beforeNaNCalls = onChange.mock.calls.length;
    fireEvent.change(screen.getByLabelText(/Count/), { target: { value: "nope" } });
    expect(onChange).toHaveBeenCalledTimes(beforeNaNCalls);

    fireEvent.click(screen.getByRole("button", { name: "enabled" }));
    expect(screen.getByText("True")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/Run Date/), { target: { value: "2026-05-01" } });
    update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update({ input: {} }).input.day).toBe("2026-05-01");

    fireEvent.click(screen.getByRole("button", { name: "Beta" }));
    update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update({ input: {} }).input.choice).toBe("beta");

    fireEvent.click(screen.getByRole("button", { name: "Value" }));
    update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update({ input: {} }).input.emptyChoice).toBe("");
  });

  it("resets the structured form from valid raw JSON and unsubscribes on unmount", () => {
    const onChange = vi.fn();

    const { unmount } = render(
      <ExecutionForm
        schema={{ type: "object", properties: { text: { type: "string" } } }}
        formData={{ input: { text: "seed" } }}
        onChange={onChange}
        validationErrors={[]}
      />
    );

    fireEvent.change(screen.getByLabelText("Raw Input JSON"), {
      target: { value: '{\n  "server": "value"\n}' },
    });

    expect(formState.resetMock).toHaveBeenLastCalledWith({ server: "value" });
    const update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update({ keep: true })).toEqual({
      keep: true,
      input: { server: "value" },
    });

    unmount();
    expect(formState.unsubscribeMock).toHaveBeenCalledTimes(1);
  });

  it("returns the previous state for equal watched values and avoids form reset for primitive JSON", () => {
    const onChange = vi.fn();

    render(
      <ExecutionForm
        schema={{ type: "object", properties: { text: { type: "string" } } }}
        formData={{ input: { text: "same" } }}
        onChange={onChange}
        validationErrors={[]}
      />
    );

    formState.emitWatch({ text: "same" });
    let update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    const previous = { input: { text: "same" }, keep: true };
    expect(update(previous)).toBe(previous);

    const resetCalls = formState.resetMock.mock.calls.length;
    fireEvent.change(screen.getByLabelText("Raw Input JSON"), {
      target: { value: '"plain-text"' },
    });

    expect(formState.resetMock).toHaveBeenCalledTimes(resetCalls);
    update = onChange.mock.calls[onChange.mock.calls.length - 1][0];
    expect(update({ keep: true })).toEqual({
      keep: true,
      input: "plain-text",
    });
  });

  it("falls back to the raw editor and warns when schema conversion fails", () => {
    const onChange = vi.fn();
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    formState.jsonSchemaToZodObjectMock.mockImplementation(() => {
      throw new Error("bad schema");
    });

    render(
      <ExecutionForm
        schema={{ type: "object" }}
        formData={{ input: { foo: "bar" } }}
        onChange={onChange}
        validationErrors={[]}
      />
    );

    expect(screen.getByText(/Structured form rendering is unavailable/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Raw Input JSON")).toHaveValue('{\n  "foo": "bar"\n}');
    expect(warnSpy).toHaveBeenCalledWith(
      "ExecutionForm: unable to convert schema to Zod",
      expect.any(Error)
    );
  });
});
