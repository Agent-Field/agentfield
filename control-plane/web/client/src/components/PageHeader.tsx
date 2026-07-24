import React from "react";
import { cn } from "@/lib/utils";
import { FilterSelect } from "./ui/FilterSelect";
import { Button, type ButtonProps } from "./ui/button";

interface PageHeaderAction {
  label: string;
  onClick: () => void;
  variant?: ButtonProps["variant"];
  size?: ButtonProps["size"];
  className?: string;
  icon?: React.ReactNode;
  disabled?: boolean;
}

export const TIME_FILTER_OPTIONS = [
  { label: "Last Hour", value: "1h" },
  { label: "Last 24 Hours", value: "24h" },
  { label: "Last 7 Days", value: "7d" },
  { label: "Last 30 Days", value: "30d" },
  { label: "All Time", value: "all" },
] as const;

export const STATUS_FILTER_OPTIONS = [
  { label: "All", value: "all" },
  { label: "Running", value: "running" },
  { label: "Paused", value: "paused" },
  { label: "Completed", value: "completed" },
  { label: "Failed", value: "failed" },
  { label: "Cancelled", value: "cancelled" },
  { label: "Pending", value: "pending" },
] as const;

interface PageHeaderFilter {
  label: string;
  value: string;
  options: ReadonlyArray<{ label: string; value: string }>;
  onChange: (value: string) => void;
}

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: PageHeaderAction[];
  filters?: PageHeaderFilter[];
  viewOptions?: React.ReactNode;
  /**
   * Additional content to render in the primary header row to the right of the title.
   * Useful for badges, toggles, or custom controls that don't fit the default actions model.
   */
  aside?: React.ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  actions = [],
  filters = [],
  viewOptions,
  aside,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn("flex flex-col gap-4", className)}>
      {/* Main Header Row */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0 flex-1">
          <h1 className="text-2xl font-semibold tracking-tight truncate">{title}</h1>
          {description && <p className="mt-1 text-sm">{description}</p>}
        </div>

        {(aside || actions.length > 0) && (
          <div className="flex flex-col gap-3 lg:ml-4 lg:flex-row lg:items-center lg:justify-end">
            {aside && (
              <div className="flex flex-wrap items-center gap-2 sm:justify-end">
                {aside}
              </div>
            )}

            {actions.length > 0 && (
              <div className="flex flex-wrap items-center gap-2 sm:justify-end">
                {actions.map((action, index) => (
                  <Button
                    key={index}
                    onClick={action.onClick}
                    disabled={action.disabled}
                    variant={action.variant}
                    size={action.size ?? "sm"}
                    className={action.className}
                  >
                    {action.icon}
                    {action.label}
                  </Button>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Filters and View Options Row */}
      {(filters.length > 0 || viewOptions) && (
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          {/* Filters */}
          {filters.length > 0 && (
            <div className="flex flex-wrap items-center gap-4">
              {filters.map((filter, index) => (
                <FilterSelect
                  key={index}
                  label={filter.label}
                  orientation="inline"
                  value={filter.value}
                  onValueChange={filter.onChange}
                  options={filter.options}
                />
              ))}
            </div>
          )}

          {/* View Options */}
          {viewOptions && (
            <div className="flex flex-wrap items-center gap-2 md:justify-end">
              {viewOptions}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
