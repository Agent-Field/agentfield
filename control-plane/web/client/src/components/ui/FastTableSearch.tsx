"use client";

import { useState, useRef, useEffect, useCallback } from "react";

import { cn } from "../../lib/utils";
import { SearchBar } from "./SearchBar";

interface FastTableSearchProps {
  onSearch: (query: string) => void;
  placeholder?: string;
  className?: string;
  resultCount?: number;
  totalCount?: number;
  disabled?: boolean;
}

export function FastTableSearch({
  onSearch,
  placeholder = "Search...",
  className,
  resultCount,
  totalCount,
  disabled = false,
}: FastTableSearchProps) {
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [shortcutHint, setShortcutHint] = useState("⌘K");

  // Debounced search with 150ms delay for optimal UX
  const debouncedSearch = useCallback(
    (() => {
      let timeoutId: NodeJS.Timeout;
      return (value: string) => {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          onSearch(value);
        }, 150);
      };
    })(),
    [onSearch]
  );

  useEffect(() => {
    debouncedSearch(query);
  }, [query, debouncedSearch]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const isMac = /mac/i.test(navigator.platform);
    setShortcutHint(isMac ? "⌘K" : "Ctrl K");
  }, []);

  const handleClear = useCallback(() => {
    setQuery("");
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd+K or Ctrl+K to focus search
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        inputRef.current?.focus();
      }
      // Escape to clear search when focused
      if (e.key === "Escape" && isFocused) {
        handleClear();
        inputRef.current?.blur();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isFocused, handleClear]);

  const showResultStats = query && resultCount !== undefined && totalCount !== undefined;

  return (
    <div className={cn("space-y-2", className)}>
      {/* Search Input */}
      <div className="relative">
        <div
          className={cn(
            "relative flex items-center rounded-md border bg-background shadow-sm transition-all duration-200",
            isFocused
              ? "border-primary ring-2 ring-primary/20"
              : "border-border hover:border-border/80",
            disabled && "cursor-not-allowed opacity-50"
          )}
        >
          <SearchBar
            ref={inputRef}
            value={query}
            onChange={setQuery}
            placeholder={placeholder}
            disabled={disabled}
            shortcutHint={shortcutHint}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            inputClassName="border-0 bg-transparent shadow-none focus-visible:ring-0 focus-visible:outline-none placeholder:text-muted-foreground"
            wrapperClassName="w-full"
          />
        </div>
      </div>

      {/* Search Results Stats */}
      {showResultStats && (
        <div className="flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <span>
              {resultCount === totalCount
                ? `${totalCount} results`
                : `${resultCount} of ${totalCount} results`}
            </span>
            {query && (
              <span>
                for "<span className="text-foreground font-medium">{query}</span>"
              </span>
            )}
          </div>

          {query && (
            <button
              onClick={() => {
                handleClear();
                inputRef.current?.focus();
              }}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      )}
    </div>
  );
}

// Search utility functions for different data types
export const createSearchMatcher = (fields: string[]) => {
  return <T extends Record<string, any>>(item: T, query: string): boolean => {
    if (!query.trim()) return true;

    const searchTerms = query.toLowerCase().split(/\s+/);

    return searchTerms.every(term => {
      return fields.some(field => {
        const value = getNestedValue(item, field);
        return value && value.toString().toLowerCase().includes(term);
      });
    });
  };
};

// Helper to get nested object values (e.g., "agent.name")
const getNestedValue = (obj: any, path: string): any => {
  return path.split('.').reduce((current, key) => current?.[key], obj);
};

// Optimized search for large datasets
export const useOptimizedSearch = <T extends Record<string, any>>(
  data: T[],
  searchQuery: string,
  searchFields: string[],
  enabled: boolean = true
) => {
  const [filteredData, setFilteredData] = useState<T[]>(data);

  useEffect(() => {
    if (!enabled) {
      setFilteredData(data);
      return;
    }

    const matcher = createSearchMatcher(searchFields);

    // Use requestIdleCallback for large datasets to avoid blocking UI
    if (data.length > 1000) {
      const filterData = () => {
        const filtered = data.filter(item => matcher(item, searchQuery));
        setFilteredData(filtered);
      };

      if ('requestIdleCallback' in window) {
        requestIdleCallback(filterData);
      } else {
        setTimeout(filterData, 0);
      }
    } else {
      // For smaller datasets, filter synchronously
      setFilteredData(data.filter(item => matcher(item, searchQuery)));
    }
  }, [data, searchQuery, searchFields, enabled]);

  return filteredData;
};
