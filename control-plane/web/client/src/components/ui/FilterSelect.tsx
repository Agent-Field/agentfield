import * as React from "react";
import { cn } from "@/lib/utils";

type Option = {
  label: string;
  value: string;
  disabled?: boolean;
};

export interface FilterSelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "onChange"> {
  label?: string;
  hideLabel?: boolean;
  options: ReadonlyArray<Option>;
  value: string;
  onValueChange: (value: string) => void;
  orientation?: "inline" | "stacked";
  helperText?: string;
  id?: string;
}

export const FilterSelect = React.forwardRef<HTMLSelectElement, FilterSelectProps>(
  (
    {
      label,
      hideLabel = false,
      options,
      value,
      onValueChange,
      orientation = "inline",
      helperText,
      id,
      className,
      disabled,
      ...rest
    },
    forwardedRef
  ) => {
    const autoId = React.useId();
    const selectId = id ?? autoId;
    const descriptionId = helperText ? `${selectId}-description` : undefined;

    const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
      onValueChange(event.target.value);
    };

    const wrapperClasses =
      orientation === "inline"
        ? "flex items-center gap-2"
        : "flex flex-col gap-1.5";

    return (
      <div className={wrapperClasses}>
        {label && !hideLabel && (
          <label
            htmlFor={selectId}
            className={cn(
              "text-sm text-muted-foreground font-medium text-muted-foreground",
              orientation === "inline" ? "whitespace-nowrap" : undefined
            )}
          >
            {label}
          </label>
        )}

        <div className="flex flex-col gap-1">
          <select
            id={selectId}
            ref={forwardedRef}
            disabled={disabled}
            value={value}
            onChange={handleChange}
            aria-describedby={descriptionId}
            className={cn(
              "flex h-9 min-w-[9.5rem] items-center rounded-md border border-border bg-muted px-3 text-sm text-foreground shadow-sm transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-0 focus:border-ring",
              "disabled:cursor-not-allowed disabled:opacity-60",
              className
            )}
            {...rest}
          >
            {options.map((option) => (
              <option key={option.value} value={option.value} disabled={option.disabled}>
                {option.label}
              </option>
            ))}
          </select>

          {helperText && (
            <span
              id={descriptionId}
              className="text-sm text-muted-foreground text-muted-foreground"
            >
              {helperText}
            </span>
          )}
        </div>
      </div>
    );
  }
);

FilterSelect.displayName = "FilterSelect";
