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

export type FilterComboboxOption = { value: string; label: string };

export interface FilterComboboxProps {
  options: ReadonlyArray<FilterComboboxOption>;
  value: string;
  onValueChange: (value: string) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  emptyMessage?: string;
  /** Accessible name for the trigger (e.g. "Time range") */
  label: string;
  className?: string;
  disabled?: boolean;
  /** Show cmdk search input; turn off for very short static lists if desired */
  searchable?: boolean;
}

export function FilterCombobox({
  options,
  value,
  onValueChange,
  placeholder = "Select…",
  searchPlaceholder = "Filter…",
  emptyMessage = "No match.",
  label,
  className,
  disabled,
  searchable = true,
}: FilterComboboxProps) {
  const [open, setOpen] = React.useState(false);
  const selected = options.find((o) => o.value === value);
  const displayLabel = selected?.label ?? placeholder;

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
            !selected && "text-muted-foreground",
            className,
          )}
        >
          <span className="truncate text-left">{displayLabel}</span>
          <ChevronsUpDown className="size-3.5 shrink-0 opacity-50" aria-hidden />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[min(100vw-2rem,var(--radix-popover-trigger-width,280px))] min-w-[12rem] max-w-[min(100vw-2rem,20rem)] p-0"
        align="start"
        sideOffset={4}
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <Command shouldFilter={searchable}>
          {searchable ? (
            <CommandInput placeholder={searchPlaceholder} className="h-9 text-xs" />
          ) : null}
          <CommandList>
            <CommandEmpty className="py-3 text-xs">{emptyMessage}</CommandEmpty>
            <CommandGroup>
              {options.map((opt) => (
                <CommandItem
                  key={opt.value}
                  value={`${opt.label} ${opt.value}`}
                  onSelect={() => {
                    onValueChange(opt.value);
                    setOpen(false);
                  }}
                  className="text-xs"
                >
                  <Check
                    className={cn(
                      "mr-2 size-3.5 shrink-0",
                      value === opt.value ? "opacity-100" : "opacity-0",
                    )}
                    aria-hidden
                  />
                  <span className="truncate">{opt.label}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
