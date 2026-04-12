import { forwardRef, useMemo, useRef, useState } from "react";
import type { InputHTMLAttributes } from "react";

import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Close, Search } from "@/components/ui/icon-bridge";

type Size = "sm" | "md";

interface SearchBarProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, "size" | "value" | "onChange"> {
  value: string;
  onChange: (value: string) => void;
  onClear?: () => void;
  size?: Size;
  shortcutHint?: string;
  wrapperClassName?: string;
  inputClassName?: string;
  clearButtonAriaLabel?: string;
}

const SIZE_STYLES: Record<
  Size,
  {
    input: string;
    icon: string;
    clearButton: string;
    clearButtonSize: string;
    shortcut: string;
  }
> = {
  sm: {
    input: "h-8 pl-8 pr-8 text-sm",
    icon: "left-2.5 h-3.5 w-3.5",
    clearButton: "right-1.5 top-1/2 -translate-y-1/2",
    clearButtonSize: "h-6 w-6",
    shortcut: "right-2.5 top-1/2 -translate-y-1/2 text-micro px-2 py-0.5",
  },
  md: {
    input: "h-10 pl-9 pr-10 text-sm",
    icon: "left-3 h-4 w-4",
    clearButton: "right-2 top-1/2 -translate-y-1/2",
    clearButtonSize: "h-7 w-7",
    shortcut: "right-3 top-1/2 -translate-y-1/2 text-xs px-2.5 py-1",
  },
};

export const SearchBar = forwardRef<HTMLInputElement, SearchBarProps>(
  (
    {
      value,
      onChange,
      onClear,
      placeholder = "Search…",
      size = "md",
      shortcutHint,
      disabled,
      wrapperClassName,
      inputClassName,
      clearButtonAriaLabel = "Clear search",
      className,
      onFocus,
      onBlur,
      ...rest
    },
    forwardedRef
  ) => {
    const localRef = useRef<HTMLInputElement>(null);
    const [isFocused, setIsFocused] = useState(false);

    const setRefs = (element: HTMLInputElement | null) => {
      localRef.current = element;
      if (typeof forwardedRef === "function") {
        forwardedRef(element);
      } else if (forwardedRef) {
        forwardedRef.current = element;
      }
    };

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
      onChange(event.target.value);
    };

    const handleClear = () => {
      if (onClear) {
        onClear();
      } else {
        onChange("");
      }
      // Keep focus on the input after clearing for faster repeated searches.
      queueMicrotask(() => localRef.current?.focus());
    };

    const handleFocus = (event: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(true);
      onFocus?.(event);
    };

    const handleBlur = (event: React.FocusEvent<HTMLInputElement>) => {
      setIsFocused(false);
      onBlur?.(event);
    };

    const sizeStyles = useMemo(() => SIZE_STYLES[size], [size]);

    return (
      <div className={cn("relative w-full", wrapperClassName)}>
        <Search
          aria-hidden="true"
          className={cn(
            "absolute top-1/2 -translate-y-1/2 text-muted-foreground",
            sizeStyles.icon
          )}
        />

        <Input
          ref={setRefs}
          value={value}
          onChange={handleChange}
          placeholder={placeholder}
          disabled={disabled}
          className={cn(
            "pr-9",
            sizeStyles.input,
            disabled ? "cursor-not-allowed opacity-60" : "",
            inputClassName,
            className
          )}
          onFocus={handleFocus}
          onBlur={handleBlur}
          {...rest}
        />

        {value && !disabled ? (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={handleClear}
            className={cn(
              "absolute text-muted-foreground hover:text-foreground",
              sizeStyles.clearButton,
              sizeStyles.clearButtonSize
            )}
            aria-label={clearButtonAriaLabel}
          >
            <Close className="h-3.5 w-3.5" />
          </Button>
        ) : null}

        {!value && shortcutHint && !isFocused && !disabled ? (
          <span
            className={cn(
              "pointer-events-none absolute select-none rounded border border-border bg-muted font-mono font-medium text-muted-foreground",
              sizeStyles.shortcut
            )}
          >
            {shortcutHint}
          </span>
        ) : null}
      </div>
    );
  }
);

SearchBar.displayName = "SearchBar";
