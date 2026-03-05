"use client";

import { useEffect, useRef, useState } from "react";
import {
  CaretDown,
  CaretRight,
  CaretUp,
  ShieldCheck,
  SpinnerGap,
} from "@/components/ui/icon-bridge";
import { useVirtualizer } from "@tanstack/react-virtual";

import { useExecutionVCStatus } from "../hooks/useVCVerification";
import { cn } from "../lib/utils";
import type { EnhancedExecution } from "../types/workflows";
import { Card, CardContent } from "./ui/card";
import { formatDurationHumanReadable, LiveElapsedDuration } from "@/components/ui/data-formatters";
import { Skeleton } from "./ui/skeleton";
import { StatusPill } from "./ui/status-pill";
import { VerifiableCredentialBadge } from "./vc/VerifiableCredentialBadge";

const COLUMN_TEMPLATE = "160px minmax(240px,1fr) 140px 110px 90px 160px 40px";
const ROW_HEIGHT = 52;

interface EnhancedExecutionsTableProps {
  executions: EnhancedExecution[];
  loading: boolean;
  hasMore: boolean;
  isFetchingMore: boolean;
  sortBy: string;
  sortOrder: "asc" | "desc";
  onSortChange: (field: string, order?: "asc" | "desc") => void;
  onLoadMore?: () => void;
  onExecutionClick?: (execution: EnhancedExecution) => void;
}

interface SortableHeaderCellProps {
  label: string;
  field: string;
  sortBy: string;
  sortOrder: "asc" | "desc";
  onSortChange: (field: string) => void;
  align?: "left" | "center" | "right";
}

function SortableHeaderCell({
  label,
  field,
  sortBy,
  sortOrder,
  onSortChange,
  align = "left",
}: SortableHeaderCellProps) {
  const isActive = sortBy === field;
  return (
    <button
      type="button"
      onClick={() => onSortChange(field)}
      className={cn(
        "flex items-center gap-1 text-xs font-semibold uppercase tracking-wide transition-colors",
        "text-muted-foreground hover:text-foreground",
        align === "right"
          ? "justify-end text-right"
          : align === "center"
            ? "justify-center text-center"
            : "justify-start text-left"
      )}
    >
      <span>{label}</span>
      <span className="flex flex-col items-center justify-center leading-none gap-0.5">
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
            "h-3 w-3",
            isActive && sortOrder === "desc"
              ? "text-primary"
              : "text-muted-foreground/40"
          )}
        />
      </span>
    </button>
  );
}

function ExecutionVCStatusCell({ executionId }: { executionId: string }) {
  const { vcStatus } = useExecutionVCStatus(executionId);

  if (!vcStatus) {
    return <VerifiableCredentialBadge hasVC={false} status="none" />;
  }

  return (
    <VerifiableCredentialBadge
      hasVC={vcStatus.has_vc}
      status={vcStatus.status}
      executionId={executionId}
      variant="table"
    />
  );
}

function LoadingRow() {
  return (
    <div
      className="absolute left-0 right-0 px-3"
      style={{ height: ROW_HEIGHT }}
    >
      <div
        className="grid items-center rounded-md bg-muted/30 px-3"
        style={{ gridTemplateColumns: COLUMN_TEMPLATE, height: ROW_HEIGHT }}
      >
        <div className="flex items-center justify-center col-span-7">
          <SpinnerGap className="h-4 w-4 animate-spin text-muted-foreground" />
          <span className="ml-2 text-body-small">
            Loading more executions…
          </span>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <Card className="card-foundation">
      <CardContent className="py-12">
        <div className="text-center space-y-2">
          <div className="text-secondary-foundation text-heading-3">
            No executions yet
          </div>
          <div className="text-tertiary-foundation">
            Executions will appear here as they run.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function EnhancedExecutionsSkeleton() {
  return (
    <Card>
      <CardContent className="p-0">
        <div className="border-b border-border px-3 py-2 bg-muted/40" />
        <div className="space-y-1 p-3">
          {Array.from({ length: 10 }).map((_, index) => (
            <div
              key={index}
              className="grid items-center rounded-lg border border-border/60 bg-muted/10 px-3 py-2"
              style={{ gridTemplateColumns: COLUMN_TEMPLATE }}
            >
              <Skeleton className="h-3 w-3 rounded-full" />
              <div className="space-y-1">
                <Skeleton className="h-3 w-48" />
                <Skeleton className="h-2.5 w-32" />
              </div>
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-3 w-16 ml-auto" />
              <Skeleton className="h-3 w-14 mx-auto" />
              <Skeleton className="h-3 w-24" />
              <Skeleton className="h-4 w-4" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function EnhancedExecutionsTable({
  executions,
  loading,
  hasMore,
  isFetchingMore,
  sortBy,
  sortOrder,
  onSortChange,
  onLoadMore,
  onExecutionClick,
}: EnhancedExecutionsTableProps) {
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: hasMore ? executions.length + 1 : executions.length,
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
    if (last.index >= executions.length - 1) {
      onLoadMore();
    }
  }, [
    virtualItems,
    onLoadMore,
    loading,
    isFetchingMore,
    hasMore,
    executions.length,
  ]);

  if (loading && executions.length === 0) {
    return <EnhancedExecutionsSkeleton />;
  }

  if (!loading && executions.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="space-y-4">
      <Card className="overflow-hidden border-border/60 shadow-sm">
        <CardContent className="p-0">
          <div className="sticky top-0 z-10 border-b border-border/60 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-3 py-2.5">
            <div
              className="grid items-center text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
              style={{ gridTemplateColumns: COLUMN_TEMPLATE }}
            >
              <SortableHeaderCell
                label="Status"
                field="status"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
              />
              <SortableHeaderCell
                label="Reasoner"
                field="task_name"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
              />
              <SortableHeaderCell
                label="Started"
                field="when"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
              />
              <SortableHeaderCell
                label="Duration"
                field="duration"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
                align="right"
              />
              <div className="flex items-center justify-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                <ShieldCheck className="h-3 w-3" /> VC
              </div>
              <SortableHeaderCell
                label="Execution ID"
                field="execution_id"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
              />
              <div className="text-right text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"></div>
            </div>
          </div>

          <div ref={parentRef} className="relative h-[60vh] overflow-y-auto bg-muted/5">
            <div
              style={{
                height: `${virtualizer.getTotalSize()}px`,
                position: "relative",
              }}
            >
              {virtualItems.map((virtualRow) => {
                const execution = executions[virtualRow.index];

                if (!execution) {
                  return <LoadingRow key={`loading-${virtualRow.key}`} />;
                }

                return (
                  <div
                    key={virtualRow.key}
                    className="absolute left-0 right-0 px-2"
                    style={{
                      transform: `translateY(${virtualRow.start}px)`,
                      height: `${virtualRow.size}px`,
                    }}
                  >
                    <div
                      className={cn(
                        "grid h-full cursor-pointer items-center rounded-md px-3 transition-all duration-200 relative overflow-hidden group",
                        "bg-card border border-transparent hover:border-border/60 hover:shadow-sm hover:bg-card"
                      )}
                      style={{ gridTemplateColumns: COLUMN_TEMPLATE }}
                      onMouseEnter={() => setHoveredRow(execution.execution_id)}
                      onMouseLeave={() => setHoveredRow(null)}
                      onClick={() => onExecutionClick?.(execution)}
                    >
                      {/* Hover Accent Bar */}
                      <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-primary opacity-0 group-hover:opacity-100 transition-opacity" />

                      <div className="flex items-center pl-2">
                        <StatusPill status={execution.status} />
                      </div>

                      <div className="flex items-center min-w-0">
                        <div className="flex flex-col min-w-0">
                          <div className="truncate font-medium text-sm">
                            {execution.task_name || "Unknown Reasoner"}
                          </div>
                          <div className="flex items-center gap-2 text-body-small">
                            <span className="inline-flex h-4 w-4 items-center justify-center rounded-full bg-muted text-[10px] font-medium">
                              {execution.agent_name
                                ? execution.agent_name.charAt(0).toUpperCase()
                                : "?"}
                            </span>
                            <span className="truncate">
                              {execution.agent_name || "Unknown"}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center">
                        <span className="text-xs text-muted-foreground font-mono">
                          {execution.relative_time}
                        </span>
                      </div>

                      <div className="flex items-center justify-end">
                        <span className="font-mono text-xs text-foreground">
                          {execution.duration_ms
                            ? formatDurationHumanReadable(execution.duration_ms)
                            : execution.status === "running" && execution.started_at
                              ? <LiveElapsedDuration startedAt={execution.started_at} className="text-blue-400" />
                              : execution.duration_display || "—"}
                        </span>
                      </div>

                      <div className="flex items-center justify-center">
                        <ExecutionVCStatusCell
                          executionId={execution.execution_id}
                        />
                      </div>

                      <div className="flex items-center">
                        <span className="font-mono text-xs text-muted-foreground group-hover:text-foreground transition-colors">
                          …{execution.execution_id.slice(-8)}
                        </span>
                      </div>

                      <div className="flex items-center justify-center">
                        <CaretRight
                          className={cn(
                            "block h-4 w-4 text-muted-foreground",
                            hoveredRow === execution.execution_id
                              ? "opacity-100"
                              : "opacity-0"
                          )}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {isFetchingMore && (
            <div className="flex items-center justify-center gap-2 border-t border-border bg-muted/20 py-3 text-body-small">
              <SpinnerGap className="h-4 w-4 animate-spin" />
              Loading more executions…
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
