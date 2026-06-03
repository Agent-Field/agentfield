import * as React from "react";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";

export interface TextInputProps
  extends React.ComponentPropsWithoutRef<typeof Input> {
  label?: string;
  hideLabel?: boolean;
  helperText?: string;
  errorText?: string;
  description?: string;
  id?: string;
}

export const TextInput = React.forwardRef<HTMLInputElement, TextInputProps>(
  (
    {
      label,
      hideLabel = false,
      helperText,
      errorText,
      description,
      id,
      className,
      ...props
    },
    forwardedRef
  ) => {
    const autoId = React.useId();
    const inputId = id ?? autoId;
    const helperId = helperText ? `${inputId}-helper` : undefined;
    const descriptionId = description ? `${inputId}-description` : undefined;
    const errorId = errorText ? `${inputId}-error` : undefined;

    const describedBy = [descriptionId, helperId, errorId]
      .filter(Boolean)
      .join(" ") || undefined;

    return (
      <div className="flex flex-col gap-1.5">
        {label && !hideLabel && (
          <label
            htmlFor={inputId}
            className="text-sm text-muted-foreground font-medium text-muted-foreground"
          >
            {label}
          </label>
        )}

        {description && (
          <p id={descriptionId} className="text-sm text-muted-foreground text-muted-foreground">
            {description}
          </p>
        )}

        <Input
          id={inputId}
          ref={forwardedRef}
          aria-describedby={describedBy}
          className={cn(
            "h-9 rounded-md border border-border bg-muted px-3 text-sm text-foreground shadow-sm transition-colors",
            "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-0",
            "disabled:cursor-not-allowed disabled:opacity-60",
            errorText && "border-status-error focus-visible:ring-status-error",
            className
          )}
          {...props}
        />

        {helperText && (
          <p id={helperId} className="text-sm text-muted-foreground text-muted-foreground">
            {helperText}
          </p>
        )}

        {errorText && (
          <p id={errorId} className="text-sm text-muted-foreground text-status-error">
            {errorText}
          </p>
        )}
      </div>
    );
  }
);

TextInput.displayName = "TextInput";
