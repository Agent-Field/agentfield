import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";
import type { JsonSchema } from "@/types/execution";
import {
  AutoForm as BaseAutoForm,
  type AutoFormFieldComponents,
  type AutoFormFieldProps,
  type AutoFormUIComponents,
} from "@autoform/react";
import { ZodProvider } from "@autoform/zod";
import type { UseFormReturn } from "react-hook-form";
import { Alert } from "../ui/alert";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Label } from "../ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../ui/select";
import { Switch } from "../ui/switch";
import { AutoExpandingTextarea } from "../ui/auto-expanding-textarea";
import { WarningFilled } from "@/components/ui/icon-bridge";
import { jsonSchemaToZodObject } from "@/utils/jsonSchemaToZod";

type FieldComponentProps = AutoFormFieldProps;
type JsonPrimitive = string | number | boolean | null;
type ExecutionFormValues = Record<string, unknown>;
type ExecutionFormWatchSubscription = ReturnType<UseFormReturn<ExecutionFormValues>["watch"]>;
type FormInputChangeEvent = React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>;
type SyntheticFieldValue = Exclude<JsonPrimitive, null>;
type SyntheticChangeEvent = {
  target: {
    name?: string;
    value: SyntheticFieldValue | "";
    checked?: SyntheticFieldValue | "";
  };
  type: "change";
};
type FormInputProps<TElement extends HTMLElement> = Record<string, unknown> & {
  ref?: React.Ref<TElement>;
  onChange?: (event: FormInputChangeEvent | SyntheticChangeEvent) => void;
  onBlur?: (event: React.FocusEvent<TElement>) => void;
  name?: string;
};

export interface ExecutionFormData extends Record<string, unknown> {
  input?: unknown;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toExecutionFormValues(value: unknown): ExecutionFormValues {
  return isRecord(value) ? value : {};
}

function assignRef<T>(ref: React.Ref<T> | undefined, value: T | null) {
  if (!ref) {
    return;
  }
  if (typeof ref === "function") {
    ref(value);
    return;
  }
  ref.current = value;
}

function normalizeInputProps<TElement extends HTMLElement>(inputProps?: FormInputProps<TElement>) {
  const { ref, onChange, onBlur, name, ...rest } = inputProps ?? {};
  return { ref, onChange, onBlur, name, rest };
}

function emitSyntheticChange(
  inputProps: Pick<FormInputProps<HTMLElement>, "onChange" | "name">,
  value: SyntheticFieldValue
) {
  if (!inputProps?.onChange) {
    return;
  }
  inputProps.onChange({
    target: {
      name: inputProps.name,
      value,
      checked: value,
    },
    type: "change",
  });
}

const uiComponents: AutoFormUIComponents = {
  Form: ({ children, className, ...props }) => (
    <form {...props} className={["space-y-4", className].filter(Boolean).join(" ")}>
      {children}
    </form>
  ),
  FieldWrapper: ({ label, error, children, id, field }) => (
    <div className="space-y-2">
      {label ? (
        <Label htmlFor={id} className="flex items-center gap-1 text-sm font-medium">
          {label}
          {field.required ? <span className="text-destructive">*</span> : null}
        </Label>
      ) : null}
      {children}
      {error ? <p className="text-sm text-destructive">{error}</p> : null}
    </div>
  ),
  ErrorMessage: ({ error }) => <p className="text-sm text-destructive">{error}</p>,
  SubmitButton: ({ children }) => (
    <Button type="submit" className="self-end">
      {children ?? "Submit"}
    </Button>
  ),
  ObjectWrapper: ({ label, children }) => (
    <div className="space-y-3 rounded-lg border border-border bg-muted/20 p-4">
      {label ? <h4 className="text-sm font-medium text-foreground">{label}</h4> : null}
      <div className="space-y-4">{children}</div>
    </div>
  ),
  ArrayWrapper: ({ label, children, onAddItem }) => (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        {label ? <span className="text-sm font-medium text-foreground">{label}</span> : null}
        <Button type="button" size="sm" variant="outline" onClick={onAddItem}>
          Add item
        </Button>
      </div>
      <div className="space-y-3">{children}</div>
    </div>
  ),
  ArrayElementWrapper: ({ children, onRemove, index }) => (
    <div className="space-y-2 rounded-md border border-border bg-muted/30 p-3">
      <div className="flex items-center justify-between text-sm text-muted-foreground">
        <span>Item {index + 1}</span>
        <Button type="button" variant="ghost" size="sm" onClick={onRemove}>
          Remove
        </Button>
      </div>
      {children}
    </div>
  ),
};

const StringField: React.FC<FieldComponentProps> = ({ id, value, inputProps }) => {
  const { ref, onChange, onBlur, name, rest } = normalizeInputProps<HTMLTextAreaElement>(inputProps);
  return (
    <AutoExpandingTextarea
      id={id}
      defaultValue={value ?? ""}
      onChange={(event) => onChange?.(event)}
      onBlur={(event) => onBlur?.(event)}
      name={name}
      ref={(element) => assignRef(ref, element)}
      maxHeight={200}
      {...rest}
    />
  );
};

const NumberField: React.FC<FieldComponentProps> = ({ id, value, inputProps }) => {
  const { ref, onChange, onBlur, name, rest } = normalizeInputProps<HTMLInputElement>(inputProps);
  const initialValue = value === undefined || value === null ? "" : String(value);
  return (
    <Input
      id={id}
      type="number"
      defaultValue={initialValue}
      onChange={(event) => {
        const next = event.target.value;
        if (next === "") {
          onChange?.({
            target: { name, value: "" },
            type: "change",
          });
          return;
        }
        const parsed = Number(next);
        if (Number.isNaN(parsed)) {
          return;
        }
        onChange?.({
          target: { name, value: parsed },
          type: "change",
        });
      }}
      onBlur={(event) => onBlur?.(event)}
      name={name}
      ref={(element) => assignRef(ref, element)}
      {...rest}
    />
  );
};

const BooleanField: React.FC<FieldComponentProps> = ({ id, value, inputProps }) => {
  const { ref, onChange, onBlur, name, rest } = normalizeInputProps<HTMLButtonElement>(inputProps);
  const [checked, setChecked] = useState(Boolean(value));

  useEffect(() => {
    setChecked(Boolean(value));
  }, [value]);

  return (
    <div className="flex items-center gap-2">
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={(next) => {
          setChecked(next);
          emitSyntheticChange({ onChange, name }, next);
        }}
        onBlur={onBlur}
        name={name}
        ref={(element) => assignRef(ref, element)}
        {...rest}
      />
      <span className="text-sm text-muted-foreground">{checked ? "True" : "False"}</span>
    </div>
  );
};

const DateField: React.FC<FieldComponentProps> = ({ id, value, inputProps }) => {
  const { ref, onChange, onBlur, name, rest } = normalizeInputProps<HTMLInputElement>(inputProps);
  const normalised =
    value instanceof Date
      ? value.toISOString().slice(0, 10)
      : typeof value === "string"
      ? value.slice(0, 10)
      : "";
  return (
    <Input
      id={id}
      type="date"
      defaultValue={normalised}
      onChange={(event) => onChange?.(event)}
      onBlur={(event) => onBlur?.(event)}
      name={name}
      ref={(element) => assignRef(ref, element)}
      {...rest}
    />
  );
};

const SelectField: React.FC<FieldComponentProps> = ({ id, value, field, inputProps }) => {
  const { onChange, name, rest } = normalizeInputProps<HTMLButtonElement>(inputProps);
  const options = field.options ?? [];
  const serialisedValue = value === undefined || value === null ? "" : String(value);
  const [selectedValue, setSelectedValue] = useState(serialisedValue);

  useEffect(() => {
    setSelectedValue(serialisedValue);
  }, [serialisedValue]);

  return (
    <Select
      value={selectedValue}
      onValueChange={(next) => {
        setSelectedValue(next);
        emitSyntheticChange({ onChange, name }, next);
      }}
      {...rest}
    >
      <SelectTrigger id={id}>
        <SelectValue placeholder="Select an option" />
      </SelectTrigger>
      <SelectContent>
        {options.length === 0 ? (
          <SelectItem value={serialisedValue || ""}>{serialisedValue || "Value"}</SelectItem>
        ) : (
          options.map(([optionValue, optionLabel]) => (
            <SelectItem key={optionValue} value={optionValue}>
              {optionLabel ?? optionValue}
            </SelectItem>
          ))
        )}
      </SelectContent>
    </Select>
  );
};

const fieldComponents: AutoFormFieldComponents = {
  string: StringField,
  number: NumberField,
  boolean: BooleanField,
  date: DateField,
  select: SelectField,
};

interface ExecutionFormProps {
  schema?: JsonSchema;
  formData: ExecutionFormData;
  onChange: Dispatch<SetStateAction<ExecutionFormData>>;
  validationErrors: string[];
}

function formatJson(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value ?? {}, null, 2);
  } catch {
    return "";
  }
}

function isDeepEqual(a: unknown, b: unknown): boolean {
  try {
    return JSON.stringify(a) === JSON.stringify(b);
  } catch {
    return false;
  }
}

export function ExecutionForm({ schema, formData, onChange, validationErrors }: ExecutionFormProps) {
  const currentInput = formData?.input;
  const initialValues = useMemo(() => toExecutionFormValues(currentInput), [currentInput]);
  const [rawJsonDraft, setRawJsonDraft] = useState<string>(() => formatJson(currentInput));
  const [rawJsonError, setRawJsonError] = useState<string | null>(null);
  const formRef = useRef<UseFormReturn<ExecutionFormValues> | null>(null);
  const skipNextInitialSyncRef = useRef(false);
  const isSyncingFromPropsRef = useRef(false);
  const subscriptionRef = useRef<ExecutionFormWatchSubscription | null>(null);

  const provider = useMemo(() => {
    if (!schema) {
      return null;
    }
    try {
      const zodSchema = jsonSchemaToZodObject(schema);
      return new ZodProvider(zodSchema);
    } catch (error) {
      console.warn("ExecutionForm: unable to convert schema to Zod", error);
      return null;
    }
  }, [schema]);

  useEffect(() => {
    if (skipNextInitialSyncRef.current) {
      skipNextInitialSyncRef.current = false;
      return;
    }

    setRawJsonDraft(formatJson(currentInput));
    setRawJsonError(null);

    const form = formRef.current;
    if (!form) {
      return;
    }

    const currentValues = form.getValues();
    if (!isDeepEqual(currentValues, initialValues)) {
      isSyncingFromPropsRef.current = true;
      form.reset(initialValues);
      isSyncingFromPropsRef.current = false;
    }
  }, [currentInput, initialValues]);

  const attachWatcher = useCallback(
    (form: UseFormReturn<ExecutionFormValues>) => {
      subscriptionRef.current?.unsubscribe?.();
      subscriptionRef.current = form.watch((values) => {
        setRawJsonDraft(formatJson(values));
        setRawJsonError(null);

        if (isSyncingFromPropsRef.current) {
          return;
        }

        skipNextInitialSyncRef.current = true;
        onChange((previous) => {
          if (isDeepEqual(previous?.input, values)) {
            return previous;
          }
          return {
            ...previous,
            input: values,
          };
        });
      });
    },
    [onChange]
  );

  const handleFormInit = useCallback(
    (form: UseFormReturn<ExecutionFormValues>) => {
      formRef.current = form;
      attachWatcher(form);
      isSyncingFromPropsRef.current = true;
      form.reset(initialValues);
      isSyncingFromPropsRef.current = false;
    },
    [attachWatcher, initialValues]
  );

  useEffect(() => {
    return () => {
      subscriptionRef.current?.unsubscribe?.();
    };
  }, []);

  const handleRawJsonChange = (value: string) => {
    setRawJsonDraft(value);
    if (!value.trim()) {
      setRawJsonError("JSON cannot be empty");
      return;
    }
    try {
      const parsed: unknown = JSON.parse(value);
      if (formRef.current && isRecord(parsed)) {
        isSyncingFromPropsRef.current = true;
        formRef.current.reset(parsed);
        isSyncingFromPropsRef.current = false;
      }
      skipNextInitialSyncRef.current = true;
      onChange((previous) => ({
        ...previous,
        input: parsed,
      }));
      setRawJsonError(null);
    } catch {
      setRawJsonError("Invalid JSON");
    }
  };

  const renderValidationAlert = () => {
    if (!validationErrors || validationErrors.length === 0) {
      return null;
    }
    return (
      <Alert className="border-red-200 bg-red-50 text-red-800">
        <WarningFilled className="h-4 w-4 text-red-600" />
        <div>
          <h4 className="font-semibold">Validation errors</h4>
          <ul className="mt-1 space-y-1 text-sm">
            {validationErrors.map((error) => (
              <li key={error}>• {error}</li>
            ))}
          </ul>
        </div>
      </Alert>
    );
  };

  const renderRawJsonEditor = () => (
    <div className="space-y-2">
      <Label htmlFor="execution-raw-json">Raw Input JSON</Label>
      <AutoExpandingTextarea
        id="execution-raw-json"
        value={rawJsonDraft}
        onChange={(event) => handleRawJsonChange(event.target.value)}
        maxHeight={320}
        className={rawJsonError ? "border-red-300" : undefined}
      />
      {rawJsonError ? (
        <p className="text-sm text-destructive">{rawJsonError}</p>
      ) : (
        <p className="text-sm text-muted-foreground">
          Edit the full JSON payload. Changes are synchronised with the structured form when the JSON
          is valid.
        </p>
      )}
    </div>
  );

  if (!provider) {
    return (
      <div className="space-y-4">
        {renderValidationAlert()}
        <Alert className="border-border bg-muted/10 text-sm text-muted-foreground">
          Structured form rendering is unavailable for this schema. Use the raw JSON editor below.
        </Alert>
        {renderRawJsonEditor()}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {renderValidationAlert()}
      <BaseAutoForm
        schema={provider}
        withSubmit={false}
        defaultValues={initialValues}
        onFormInit={handleFormInit}
        uiComponents={uiComponents}
        formComponents={fieldComponents}
      />
      <details className="mt-4 rounded-md border border-border bg-muted/20">
        <summary className="cursor-pointer select-none px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground">
          Advanced: Edit raw JSON
        </summary>
        <div className="space-y-2 border-t border-border px-4 py-3">{renderRawJsonEditor()}</div>
      </details>
    </div>
  );
}
