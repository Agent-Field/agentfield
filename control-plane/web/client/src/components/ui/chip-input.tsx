import { useState, useRef, useEffect, useCallback } from "react";
import { X } from "@/components/ui/icon-bridge";
import { Badge } from "./badge";
import { cn } from "@/lib/utils";

interface ChipInputProps {
  value: string[];
  onChange: (tags: string[]) => void;
  suggestions?: string[];
  placeholder?: string;
  className?: string;
}

export function ChipInput({
  value,
  onChange,
  suggestions = [],
  placeholder = "Type and press Enter...",
  className,
}: ChipInputProps) {
  const [inputValue, setInputValue] = useState("");
  const [showDropdown, setShowDropdown] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter suggestions: exclude already-selected tags and match typed text
  const filtered = suggestions.filter(
    (s) =>
      !value.includes(s) &&
      (inputValue === "" || s.toLowerCase().includes(inputValue.toLowerCase()))
  );

  const addChip = useCallback(
    (tag: string) => {
      const trimmed = tag.trim();
      if (trimmed && !value.includes(trimmed)) {
        onChange([...value, trimmed]);
      }
      setInputValue("");
      setHighlightIndex(-1);
    },
    [value, onChange]
  );

  const removeChip = useCallback(
    (tag: string) => {
      onChange(value.filter((t) => t !== tag));
    },
    [value, onChange]
  );

  // Close dropdown on outside click
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(e.target as Node)
      ) {
        setShowDropdown(false);
        setHighlightIndex(-1);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      if (highlightIndex >= 0 && highlightIndex < filtered.length) {
        addChip(filtered[highlightIndex]);
      } else if (inputValue.trim()) {
        addChip(inputValue);
      }
    } else if (
      e.key === "Backspace" &&
      inputValue === "" &&
      value.length > 0
    ) {
      removeChip(value[value.length - 1]);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setHighlightIndex((prev) =>
        prev < filtered.length - 1 ? prev + 1 : prev
      );
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setHighlightIndex((prev) => (prev > 0 ? prev - 1 : -1));
    } else if (e.key === "Escape") {
      setShowDropdown(false);
      setHighlightIndex(-1);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
    setHighlightIndex(-1);
    if (!showDropdown) {
      setShowDropdown(true);
    }
  };

  const shouldShowDropdown = showDropdown && filtered.length > 0;

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <div
        className={cn(
          "flex flex-wrap items-center gap-1.5 rounded-md border border-input bg-transparent px-2.5 py-1.5 text-sm",
          "ring-offset-background focus-within:ring-2 focus-within:ring-ring focus-within:ring-offset-2",
          "min-h-9 cursor-text"
        )}
        onClick={() => inputRef.current?.focus()}
      >
        {value.map((tag) => (
          <Badge
            key={tag}
            variant="secondary"
            className="gap-1 pr-1 shrink-0"
            showIcon={false}
          >
            {tag}
            <button
              type="button"
              className="ml-0.5 rounded-sm opacity-70 hover:opacity-100 focus:outline-none"
              onClick={(e) => {
                e.stopPropagation();
                removeChip(tag);
              }}
              aria-label={`Remove ${tag}`}
            >
              <X size={12} />
            </button>
          </Badge>
        ))}
        <input
          ref={inputRef}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          onFocus={() => setShowDropdown(true)}
          placeholder={value.length === 0 ? placeholder : ""}
          className="flex-1 min-w-20 bg-transparent outline-none text-sm placeholder:text-muted-foreground"
        />
      </div>

      {shouldShowDropdown && (
        <div className="absolute z-50 mt-1 w-full rounded-md border bg-popover shadow-md max-h-48 overflow-y-auto">
          {filtered.map((suggestion, idx) => (
            <button
              key={suggestion}
              type="button"
              className={cn(
                "w-full text-left px-3 py-2 text-sm hover:bg-accent hover:text-accent-foreground transition-colors",
                idx === highlightIndex && "bg-accent text-accent-foreground"
              )}
              onMouseDown={(e) => {
                e.preventDefault(); // prevent blur before click registers
                addChip(suggestion);
              }}
              onMouseEnter={() => setHighlightIndex(idx)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
