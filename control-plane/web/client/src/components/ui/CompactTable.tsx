"use client";

import { useEffect, useRef, useState } from "react";
import {
  CaretDown,
  CaretRight,
  CaretUp,
  SpinnerGap,
} from "@/components/ui/icon-bridge";
import { useVirtualizer } from "@tanstack/react-virtual";
import { cn } from "../../lib/utils";
import { Skeleton } from "./skeleton";
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "./empty";
import { Button } from "./button";

const DEFAULT_ROW_HEIGHT = 32; // Using foundation's compact row height
const CHEVRON_COLUMN_WIDTH = "28px";

interface SortableHeaderCellProps {
  label: string;
  field: string;
  sortBy: string;
  sortOrder: "asc" | "desc";
  onSortChange: (field: string) => void;
  align?: "left" | "center" | "right";
  className?: string;
}

function SortableHeaderCell({
  label,
  field,
  sortBy,
  sortOrder,
  onSortChange,
  align = "left",
  className,
}: SortableHeaderCellProps) {
  const isActive = sortBy === field;
  return (
    <button
          type="button"
          onClick={() => onSortChange(field)}
          className={cn(
            "flex w-full min-w-0 items-center gap-1 whitespace-nowrap overflow-hidden text-xs font-medium uppercase tracking-wide transition-colors",
            "text-muted-foreground hover:text-foreground foundation-transition",
            align === "right"
              ? "justify-end text-right"
              : align === "center"
                ? "justify-center text-center"
                : "justify-start text-left",
            className
          )}
        >
      <span className="truncate">{label}</span>
      <span className="flex flex-col leading-none">
        <CaretUp
          className={cn(
            "h-3 w-3",
            isActive && sortOrder === "asc"
              ? "text-primary"
              : "text-muted-foreground/40"
          )}
        />
        <CaretDown
          className={cn(
            "h-3 w-3 -mt-1",
            isActive && sortOrder === "desc"
              ? "text-primary"
              : "text-muted-foreground/40"
          )}
        />
      </span>
    </button>
  );
}

function StaticHeaderCell({
  label,
  align = "left",
  className,
  children,
}: {
  label?: string;
  align?: "left" | "center" | "right";
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "text-xs font-medium uppercase tracking-wider text-muted-foreground/80 truncate whitespace-nowrap min-w-0",
        align === "right"
          ? "text-right"
          : align === "center"
            ? "text-center"
            : "text-left",
        className
      )}
    >
      {children || label}
    </div>
  );
}

interface CompactTableProps<T> {
  data: T[];
  loading: boolean;
  hasMore: boolean;
  isFetchingMore: boolean;
  sortBy: string;
  sortOrder: "asc" | "desc";
  onSortChange: (field: string, order?: "asc" | "desc") => void;
  onLoadMore?: () => void;
  onRowClick?: (item: T) => void;
  columns: Array<{
    key: string;
    header: string | React.ReactNode;
    sortable?: boolean;
    align?: "left" | "center" | "right";
    width?: string;
    render: (item: T, index: number) => React.ReactNode;
  }>;
  gridTemplate: string;
  emptyState?: {
    title: string;
    description: string;
    icon?: React.ReactNode;
    action?: {
      label: string;
      onClick: () => void;
      icon?: React.ReactNode;
    };
    secondaryAction?: {
      label: string;
      onClick: () => void;
      icon?: React.ReactNode;
    };
  };
  className?: string;
  getRowKey: (item: T) => string;
  rowHeight?: number;
}

function LoadingRow({ gridTemplate, rowHeight }: { gridTemplate: string; rowHeight: number }) {
  return (
    <div
      className="absolute left-0 right-0"
      style={{ height: rowHeight }}
    >
      <div
        className="grid items-center h-full px-3 bg-muted/20 rounded-sm"
        style={{ gridTemplateColumns: gridTemplate }}
      >
        <div className="flex items-center justify-center col-span-full">
          <SpinnerGap className="h-3 w-3 animate-spin text-muted-foreground" />
          <span className="ml-2 text-sm text-muted-foreground">
            Loading more…
          </span>
        </div>
      </div>
    </div>
  );
}

function EmptyState({
  title,
  description,
  icon,
  action,
  secondaryAction,
}: {
  title: string;
  description: string;
  icon?: React.ReactNode;
  action?: { label: string; onClick: () => void; icon?: React.ReactNode };
  secondaryAction?: {
    label: string;
    onClick: () => void;
    icon?: React.ReactNode;
  };
}) {
  return (
    <Empty className="border-none bg-transparent py-10 shadow-none">
      <EmptyHeader>
        {icon ? (
          <EmptyMedia variant="icon">{icon}</EmptyMedia>
        ) : null}
        <EmptyTitle>{title}</EmptyTitle>
        <EmptyDescription>{description}</EmptyDescription>
      </EmptyHeader>
      {(action || secondaryAction) && (
        <EmptyContent>
          {action ? (
            <Button onClick={action.onClick} className="inline-flex items-center gap-2">
              {action.icon}
              {action.label}
            </Button>
          ) : null}
          {secondaryAction ? (
            <Button
              variant="outline"
              onClick={secondaryAction.onClick}
              className="inline-flex items-center gap-2"
            >
              {secondaryAction.icon}
              {secondaryAction.label}
            </Button>
          ) : null}
        </EmptyContent>
      )}
    </Empty>
  );
}

function CompactTableSkeleton({
  gridTemplate,
  columnCount = 6,
  rowCount = 10,
}: {
  gridTemplate: string;
  columnCount?: number;
  rowCount?: number;
}) {
  return (
    <div className="space-y-0.5">
      {Array.from({ length: rowCount }).map((_, index) => (
        <div
          key={index}
          className="grid items-center h-8 px-3 bg-muted/10 rounded-sm"
          style={{ gridTemplateColumns: gridTemplate }}
        >
          {Array.from({ length: columnCount }).map((_, colIndex) => (
            <Skeleton
              key={colIndex}
              className="h-2 rounded"
              style={{ width: `${60 + Math.random() * 40}%` }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

export function CompactTable<T>({
  data,
  loading,
  hasMore,
  isFetchingMore,
  sortBy,
  sortOrder,
  onSortChange,
  onLoadMore,
  onRowClick,
  columns,
  gridTemplate,
  emptyState = {
    title: "No data",
    description: "Data will appear here when available.",
  },
  className,
  getRowKey,
  rowHeight: rowHeightProp,
}: CompactTableProps<T>) {
  const ROW_HEIGHT = rowHeightProp ?? DEFAULT_ROW_HEIGHT;
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const parentRef = useRef<HTMLDivElement>(null);

  const hasRowClick = Boolean(onRowClick);
  const resolvedGridTemplate = hasRowClick
    ? `${gridTemplate} ${CHEVRON_COLUMN_WIDTH}`
    : gridTemplate;

  const virtualizer = useVirtualizer({
    count: hasMore ? data.length + 1 : data.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 8,
  });

  const virtualItems = virtualizer.getVirtualItems();

  useEffect(() => {
    if (!onLoadMore || loading || isFetchingMore || !hasMore) {
      return;
    }
    const last = virtualItems[virtualItems.length - 1];
    if (!last) {
      return;
    }
    if (last.index >= data.length - 1) {
      onLoadMore();
    }
  }, [virtualItems, onLoadMore, loading, isFetchingMore, hasMore, data.length]);

  if (loading && data.length === 0) {
    return (
      <div className={cn("space-y-4", className)}>
        <div className="bg-card border-0 rounded-md shadow-sm">
          <div className="border-b border-border/50 px-3 py-2 bg-muted/30">
            <div
              className="grid items-center"
              style={{ gridTemplateColumns: resolvedGridTemplate }}
            >
              {columns.map((_, index) => (
                <Skeleton key={index} className="h-2 rounded" />
              ))}
              {hasRowClick && <div className="h-2" />}
            </div>
          </div>
          <div className="p-3">
            <CompactTableSkeleton
              gridTemplate={resolvedGridTemplate}
              columnCount={columns.length + (hasRowClick ? 1 : 0)}
            />
          </div>
        </div>
      </div>
    );
  }

  if (!loading && data.length === 0) {
    return (
      <div className={cn("space-y-4", className)}>
        <div className="bg-card border-0 rounded-md shadow-sm">
          <EmptyState
            title={emptyState.title}
            description={emptyState.description}
            icon={emptyState.icon}
            action={emptyState.action}
            secondaryAction={emptyState.secondaryAction}
          />
        </div>
      </div>
    );
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div className="bg-card border-0 rounded-md shadow-sm overflow-hidden">
        {/* Header */}
        <div className="border-b border-border/50 px-3 py-2 bg-muted/30">
          <div
            className="grid items-center"
            style={{ gridTemplateColumns: resolvedGridTemplate }}
          >
            {columns.map((column, index) => (
              <div key={index} className="min-w-0 overflow-hidden px-1">
                {column.sortable ? (
                  <SortableHeaderCell
                    label={typeof column.header === 'string' ? column.header : ''}
                    field={column.key}
                    sortBy={sortBy}
                    sortOrder={sortOrder}
                    onSortChange={onSortChange}
                    align={column.align}
                  />
                ) : (
                  <StaticHeaderCell align={column.align}>
                    {column.header}
                  </StaticHeaderCell>
                )}
              </div>
            ))}
            {hasRowClick && <div className="px-1" aria-hidden />}
          </div>
        </div>

        {/* Virtual Table Body */}
        <div ref={parentRef} className="relative h-[60vh] overflow-y-auto">
          <div
            style={{
              height: `${virtualizer.getTotalSize()}px`,
              position: "relative",
            }}
          >
            {virtualItems.map((virtualRow) => {
              const item = data[virtualRow.index];

              if (!item) {
                return (
                  <LoadingRow
                    key={`loading-${virtualRow.key}`}
                    gridTemplate={resolvedGridTemplate}
                    rowHeight={ROW_HEIGHT}
                  />
                );
              }

              const rowKey = getRowKey(item);

              return (
                <div
                  key={virtualRow.key}
                  className="absolute left-0 right-0"
                  style={{
                    transform: `translateY(${virtualRow.start}px)`,
                    height: `${virtualRow.size}px`,
                  }}
                >
                  <div
                    className={cn(
                      "grid h-full items-center content-center px-4 transition-colors duration-150 border-l-2 border-transparent foundation-transition",
                      "hover:bg-[var(--row-hover-bg)] dark:hover:bg-[var(--dark-row-hover-bg)] hover:border-l-muted-foreground/30",
                      hasRowClick &&
                        "cursor-pointer active:bg-[hsl(var(--accent))] dark:active:bg-[var(--dark-row-hover-bg)]"
                    )}
                    style={{ gridTemplateColumns: resolvedGridTemplate }}
                    onMouseEnter={() => setHoveredRow(rowKey)}
                    onMouseLeave={() => setHoveredRow(null)}
                    onClick={() => onRowClick?.(item)}
                  >
                    {columns.map((column, index) => (
                      <div
                        key={index}
                        className={cn(
                          "min-w-0 px-1.5 flex h-full items-center",
                          column.align === "right"
                            ? "justify-end"
                            : column.align === "center"
                              ? "justify-center"
                              : "justify-start"
                        )}
                      >
                        {column.render(item, virtualRow.index)}
                      </div>
                    ))}

                    {/* Row Chevron */}
                    {hasRowClick && (
                      <div className="flex h-full items-center justify-end px-1.5">
                        <CaretRight
                          className={cn(
                            "h-3 w-3 text-muted-foreground transition-opacity",
                            hoveredRow === rowKey ? "opacity-100" : "opacity-0"
                          )}
                        />
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Loading More Footer */}
        {isFetchingMore && (
          <div className="flex items-center justify-center gap-2 border-t border-border/50 bg-muted/20 py-2">
            <SpinnerGap className="h-3 w-3 animate-spin" />
            <span className="text-sm text-muted-foreground">
              Loading more…
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// Export types for reuse
export type { CompactTableProps, SortableHeaderCellProps };
export { SortableHeaderCell, StaticHeaderCell };
