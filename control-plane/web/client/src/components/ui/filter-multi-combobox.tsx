import * as React from "react";
import { Check, ChevronsUpDown } from "lucide-react";

import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

export type MultiFilterOption = {
  value: string;
  label: string;
  /** Shown before the label (e.g. status color dot) */
  leading?: React.ReactNode;
};

export interface FilterMultiComboboxProps {
  options: ReadonlyArray<MultiFilterOption>;
  /** Empty set means “no restriction” (e.g. all statuses / all agents). */
  selected: Set<string>;
  /** Functional updates avoid stale closures when toggling quickly. */
  onSelectedChange: (updater: (prev: Set<string>) => Set<string>) => void;
  /** Trigger text when `selected` is empty */
  emptyLabel: string;
  searchPlaceholder?: string;
  emptyMessage?: string;
  /** Accessible name for the trigger */
  label: string;
  className?: string;
  /** e.g. (n) => `${n} statuses` */
  pluralLabel: (count: number) => string;
  disabled?: boolean;
}

function toggleSet(set: Set<string>, value: string): Set<string> {
  const next = new Set(set);
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }
  return next;
}

function triggerSummary(
  options: ReadonlyArray<MultiFilterOption>,
  selected: Set<string>,
  emptyLabel: string,
  pluralLabel: (count: number) => string,
): string {
  if (selected.size === 0) return emptyLabel;
  if (selected.size === 1) {
    const v = [...selected][0];
    return options.find((o) => o.value === v)?.label ?? v;
  }
  return pluralLabel(selected.size);
}

export function FilterMultiCombobox({
  options,
  selected,
  onSelectedChange,
  emptyLabel,
  searchPlaceholder = "Filter…",
  emptyMessage = "No match.",
  label,
  className,
  pluralLabel,
  disabled,
}: FilterMultiComboboxProps) {
  const [open, setOpen] = React.useState(false);
  const summary = triggerSummary(options, selected, emptyLabel, pluralLabel);
  const isFiltered = selected.size > 0;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          aria-label={label}
          disabled={disabled}
          className={cn(
            "h-9 justify-between gap-1.5 px-3 font-normal shadow-xs",
            "min-w-[9.5rem] max-w-[14rem] text-xs sm:min-w-[10rem]",
            !isFiltered && "text-muted-foreground",
            className,
          )}
        >
          <span className="truncate text-left">{summary}</span>
          <ChevronsUpDown className="size-3.5 shrink-0 opacity-50" aria-hidden />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[min(100vw-2rem,var(--radix-popover-trigger-width,280px))] min-w-[13rem] max-w-[min(100vw-2rem,22rem)] p-0"
        align="start"
        sideOffset={4}
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <Command>
          <CommandInput placeholder={searchPlaceholder} className="h-9 text-xs" />
          <CommandList>
            <CommandEmpty className="py-3 text-xs">{emptyMessage}</CommandEmpty>
            <CommandGroup className="max-h-[min(50vh,280px)] overflow-y-auto p-1">
              {options.map((opt) => {
                const checked = selected.has(opt.value);
                return (
                  <CommandItem
                    key={opt.value}
                    value={`${opt.label} ${opt.value}`}
                    onSelect={() => {
                      onSelectedChange((prev) => toggleSet(prev, opt.value));
                    }}
                    className="cursor-pointer text-xs"
                  >
                    <span
                      className={cn(
                        "mr-2 flex size-4 shrink-0 items-center justify-center rounded-sm border border-primary/60",
                        checked
                          ? "border-primary bg-primary text-primary-foreground"
                          : "bg-background opacity-80",
                      )}
                      aria-hidden
                    >
                      {checked ? <Check className="size-3" strokeWidth={3} aria-hidden /> : null}
                    </span>
                    {opt.leading ? (
                      <span className="mr-2 flex shrink-0 items-center">{opt.leading}</span>
                    ) : null}
                    <span className="flex-1 truncate">{opt.label}</span>
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
        {selected.size > 0 ? (
          <div className="border-t border-border p-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-8 w-full text-xs text-muted-foreground hover:text-foreground"
              onClick={() => onSelectedChange(() => new Set())}
            >
              Clear selection
            </Button>
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}
