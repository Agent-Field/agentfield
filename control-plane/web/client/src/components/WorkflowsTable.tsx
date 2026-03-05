"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  CaretDown,
  CaretRight,
  CaretUp,
  ShieldCheck,
  Trash,
  SpinnerGap,
} from "@/components/ui/icon-bridge";
import { useVirtualizer } from "@tanstack/react-virtual";

import { useWorkflowVCStatuses } from "../hooks/useVCVerification";
import { statusTone } from "../lib/theme";
import { cn } from "../lib/utils";
import type { VCStatusSummary } from "../types/did";
import type { WorkflowSummary } from "../types/workflows";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { Checkbox } from "./ui/checkbox";
import { formatDurationHumanReadable, LiveElapsedDuration } from "@/components/ui/data-formatters";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "./ui/hover-card";
import { Skeleton } from "./ui/skeleton";
import { StatusPill } from "./ui/status-pill";
import { VerifiableCredentialBadge } from "./vc/VerifiableCredentialBadge";
import { WorkflowDeleteDialog } from "./workflows/WorkflowDeleteDialog";
import {
  deleteWorkflows,
  type WorkflowCleanupResult,
} from "../services/workflowsApi";
import { normalizeExecutionStatus } from "../utils/status";

const COLUMN_TEMPLATE =
  "40px 160px minmax(220px,1fr) 200px 90px 90px 80px 120px 72px 72px";
const ROW_HEIGHT = 56;
const LOAD_MORE_THRESHOLD = 5;

interface WorkflowsTableProps {
  workflows: WorkflowSummary[];
  loading: boolean;
  hasMore: boolean;
  isFetchingMore: boolean;
  sortBy: string;
  sortOrder: "asc" | "desc";
  onSortChange: (field: string, order?: "asc" | "desc") => void;
  onLoadMore?: () => void;
  onWorkflowClick?: (workflow: WorkflowSummary) => void;
  onWorkflowsDeleted?: () => void;
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
        "text-muted-foreground hover:text-foreground", // default colors
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

function RelativeTime({ timestamp }: { timestamp: string }) {
  const getRelativeTime = (value: string) => {
    const now = new Date();
    const time = new Date(value);
    const diffMs = now.getTime() - time.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return "now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  return (
    <span className="timestamp-foundation">{getRelativeTime(timestamp)}</span>
  );
}

const getStatusCounts = (workflow: WorkflowSummary): Record<string, number> => {
  return workflow.status_counts ?? {};
};

const getActiveExecutions = (workflow: WorkflowSummary): number => {
  return workflow.active_executions ?? 0;
};

const getFailedExecutions = (workflow: WorkflowSummary): number => {
  const counts = getStatusCounts(workflow);
  return (counts.failed ?? 0) + (counts.timeout ?? 0);
};

function WorkflowHoverCard({
  workflow,
  children,
}: {
  workflow: WorkflowSummary;
  children: ReactNode;
}) {
  const statusCounts = getStatusCounts(workflow);
  const activeExecutions = getActiveExecutions(workflow);
  const agentLabel = workflow.agent_name ?? workflow.agent_id ?? '—';
  const failedExecutions = getFailedExecutions(workflow);

  return (
    <HoverCard>
      <HoverCardTrigger asChild>{children}</HoverCardTrigger>
      <HoverCardContent className="w-80 p-4">
        <div className="space-y-3">
          <div>
            <h4 className="font-semibold text-sm mb-1">
              {workflow.display_name}
            </h4>
            <p className="text-body-small font-mono">
              ID: {workflow.workflow_id}
            </p>
            <p className="text-body-small text-muted-foreground font-mono">
              Run: {workflow.run_id}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="text-muted-foreground">Agent:</span>
              <div className="font-medium">{agentLabel}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Root Task:</span>
              <div className="font-medium">{workflow.root_reasoner}</div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <span className="text-muted-foreground">Executions:</span>
              <div className="font-medium">{workflow.total_executions}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Max Depth:</span>
              <div className="font-medium">{workflow.max_depth}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Status:</span>
              <div className="font-medium capitalize">{workflow.status}</div>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <span className="text-muted-foreground">Active:</span>
              <div className="font-medium">{activeExecutions}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Succeeded:</span>
              <div className="font-medium">{statusCounts.succeeded ?? 0}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Failed:</span>
              <div className="font-medium text-destructive">{failedExecutions}</div>
            </div>
          </div>

          {workflow.session_id && (
            <div className="text-xs">
              <span className="text-muted-foreground">Session:</span>
              <div className="font-mono text-xs">
                {workflow.session_id.slice(0, 16)}...
              </div>
            </div>
          )}
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

function WorkflowVCStatusCell({
  workflowId,
  summary,
  loading,
}: {
  workflowId: string;
  summary?: VCStatusSummary | null;
  loading?: boolean;
}) {
  if (loading && !summary) {
    return <Skeleton className="h-4 w-16 rounded-full bg-muted/20" />;
  }

  if (!summary) {
    return (
      <VerifiableCredentialBadge
        hasVC={false}
        status="none"
        workflowId={workflowId}
        variant="table"
      />
    );
  }

  return (
    <VerifiableCredentialBadge
      hasVC={summary.has_vcs}
      status={summary.verification_status}
      workflowId={workflowId}
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
        <div className="flex items-center justify-center">
          <SpinnerGap className="h-4 w-4 animate-spin text-muted-foreground" />
        </div>
        <div className="col-span-9 text-body-small">
          Loading more workflows…
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
            No workflows yet
          </div>
          <div className="text-tertiary-foundation">
            Workflows will appear here as they execute.
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function WorkflowsTableSkeleton() {
  return (
    <Card>
      <CardContent className="p-0">
        <div className="border-b border-border px-3 py-2 bg-muted/40" />
        <div className="space-y-1 p-3">
          {Array.from({ length: 8 }).map((_, index) => (
            <div
              key={index}
              className="grid items-center rounded-lg border border-border/60 bg-muted/10 px-3 py-2"
              style={{ gridTemplateColumns: COLUMN_TEMPLATE }}
            >
              <Skeleton className="h-4 w-4" />
              <Skeleton className="h-3 w-3 rounded-full" />
              <div className="space-y-1">
                <Skeleton className="h-3 w-40" />
                <Skeleton className="h-2 w-24" />
              </div>
              <Skeleton className="h-3 w-32" />
              <Skeleton className="h-3 w-12 ml-auto" />
              <Skeleton className="h-3 w-16 ml-auto" />
              <Skeleton className="h-3 w-16 ml-auto" />
              <Skeleton className="h-3 w-16 mx-auto" />
              <Skeleton className="h-3 w-20 ml-auto" />
              <div className="flex items-center justify-center gap-2">
                <Skeleton className="h-6 w-6 rounded" />
                <Skeleton className="h-4 w-4 rounded" />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export function WorkflowsTable({
  workflows,
  loading,
  hasMore,
  isFetchingMore,
  sortBy,
  sortOrder,
  onSortChange,
  onLoadMore,
  onWorkflowClick,
  onWorkflowsDeleted,
}: WorkflowsTableProps) {
  const [selectedWorkflows, setSelectedWorkflows] = useState<Set<string>>(
    new Set()
  );
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);

  const parentRef = useRef<HTMLDivElement>(null);

  const workflowIds = useMemo(
    () => workflows.map((workflow) => workflow.workflow_id).filter(Boolean),
    [workflows]
  );
  const { statuses: workflowVCStatuses, loading: workflowVCStatusesLoading } =
    useWorkflowVCStatuses(workflowIds);

  useEffect(() => {
    // Keep selection in sync with loaded workflows
    setSelectedWorkflows((prev) => {
      const next = new Set<string>();
      workflows.forEach((workflow) => {
        if (prev.has(workflow.run_id)) {
          next.add(workflow.run_id);
        }
      });
      return next;
    });
  }, [workflows]);

  const virtualizer = useVirtualizer({
    count: hasMore ? workflows.length + 1 : workflows.length,
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
    const thresholdIndex = Math.max(workflows.length - LOAD_MORE_THRESHOLD, 0);
    if (last.index >= thresholdIndex) {
      onLoadMore();
    }
  }, [
    virtualItems,
    onLoadMore,
    loading,
    isFetchingMore,
    hasMore,
    workflows.length,
  ]);

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedWorkflows(new Set(workflows.map((wf) => wf.run_id)));
    } else {
      setSelectedWorkflows(new Set());
    }
  };

  const handleSelectWorkflow = (workflowRunId: string, checked: boolean) => {
    setSelectedWorkflows((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(workflowRunId);
      } else {
        next.delete(workflowRunId);
      }
      return next;
    });
  };

  const handleDeleteSelected = async (
    workflowIds: string[]
  ): Promise<WorkflowCleanupResult[]> => {
    const results = await deleteWorkflows(workflowIds);

    if (results.length === 0 && workflowIds.length > 0) {
      throw new Error('Failed to delete workflows. Please try again.');
    }

    const failedResults = results.filter((result) => !result.success);
    if (failedResults.length > 0) {
      const message =
        failedResults.find((result) => result.error_message)?.error_message ||
        `Failed to delete ${failedResults.length} workflow${failedResults.length === 1 ? '' : 's'}.`;
      throw new Error(message);
    }

    setSelectedWorkflows(new Set());
    setShowDeleteDialog(false);
    onWorkflowsDeleted?.();

    return results;
  };

  const selectedWorkflowObjects = useMemo(
    () => workflows.filter((wf) => selectedWorkflows.has(wf.run_id)),
    [workflows, selectedWorkflows]
  );

  const isAllSelected =
    workflows.length > 0 && selectedWorkflows.size === workflows.length;
  const isPartiallySelected =
    selectedWorkflows.size > 0 && selectedWorkflows.size < workflows.length;

  if (loading && workflows.length === 0) {
    return <WorkflowsTableSkeleton />;
  }

  if (!loading && workflows.length === 0) {
    return <EmptyState />;
  }

  return (
    <div className="space-y-4">
      {selectedWorkflows.size > 0 && (
        <Card className="border-accent-primary/20 bg-accent-primary/5">
          <CardContent className="py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span className="text-sm font-medium">
                  {selectedWorkflows.size} workflow
                  {selectedWorkflows.size === 1 ? "" : "s"} selected
                </span>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setSelectedWorkflows(new Set())}
                >
                  Clear Selection
                </Button>
              </div>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => setShowDeleteDialog(true)}
                className="flex items-center gap-2"
              >
                <Trash className="w-4 h-4" />
                Delete Selected
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent className="p-0">
          <div className="border-b border-border px-3 py-2 bg-muted/40">
            <div
              className="grid items-center text-xs font-semibold uppercase tracking-wide text-muted-foreground"
              style={{ gridTemplateColumns: COLUMN_TEMPLATE }}
            >
              <div className="flex items-center justify-center">
                <Checkbox
                  checked={isAllSelected}
                  onCheckedChange={(value) => handleSelectAll(Boolean(value))}
                  aria-label="Select all workflows"
                  className={cn(
                    isPartiallySelected && "data-[state=checked]:bg-primary/50"
                  )}
                />
              </div>
              <SortableHeaderCell
                label="Status"
                field="status"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
              />
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Workflow
              </div>
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Last Reasoner
              </div>
              <SortableHeaderCell
                label="Nodes"
                field="total_executions"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
                align="right"
              />
              <div className="text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Active
              </div>
              <div className="text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Duration
              </div>
              <div className="flex items-center justify-center gap-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                <ShieldCheck className="h-3 w-3" /> VC
              </div>
              <SortableHeaderCell
                label="Updated"
                field="latest_activity"
                sortBy={sortBy}
                sortOrder={sortOrder}
                onSortChange={onSortChange}
                align="right"
              />
              <div className="text-right text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Actions
              </div>
            </div>
          </div>

          <div ref={parentRef} className="relative h-[60vh] overflow-y-auto">
            <div
              style={{
                height: `${virtualizer.getTotalSize()}px`,
                position: "relative",
              }}
            >
              {virtualItems.map((virtualRow) => {
                const workflow = workflows[virtualRow.index];

                if (!workflow) {
                  return <LoadingRow key={`loading-${virtualRow.key}`} />;
                }

                const normalizedStatus = normalizeExecutionStatus(workflow.status);
                const activeCount = getActiveExecutions(workflow);
                const issueCount = getFailedExecutions(workflow);

                const isSelected = selectedWorkflows.has(workflow.run_id);

                return (
                  <div
                    key={virtualRow.key}
                    className={cn(
                      "absolute left-0 right-0 px-3", // positioning
                      "transition-colors",
                      isSelected &&
                        "bg-accent-primary/10 border border-accent-primary/30 rounded-md"
                    )}
                    style={{
                      transform: `translateY(${virtualRow.start}px)`,
                      height: `${virtualRow.size}px`,
                    }}
                    onMouseEnter={() => setHoveredRow(workflow.run_id)}
                    onMouseLeave={() => setHoveredRow(null)}
                    onClick={() => onWorkflowClick?.(workflow)}
                  >
                    <div
                      className={cn(
                        "grid h-full items-center rounded-lg px-3", // layout
                        "hover:bg-[var(--row-hover-bg)] dark:hover:bg-[var(--dark-row-hover-bg)]",
                        isSelected ? "bg-accent-primary/10" : "bg-card"
                      )}
                      style={{ gridTemplateColumns: COLUMN_TEMPLATE }}
                    >
                      <div className="flex items-center justify-center">
                        <div onClick={(event) => event.stopPropagation()}>
                          <Checkbox
                            checked={isSelected}
                            onCheckedChange={(value) =>
                              handleSelectWorkflow(
                                workflow.run_id,
                                Boolean(value)
                              )
                            }
                            aria-label={`Select workflow ${workflow.display_name}`}
                          />
                        </div>
                      </div>

                      <div className="flex items-center gap-3">
                        <StatusPill status={normalizedStatus} />
                        {(activeCount > 0 || issueCount > 0) && (
                          <div className="flex items-center gap-2 text-body-small">
                            {activeCount > 0 && (
                              <span className={cn(statusTone.info.accent, "font-medium")}>{activeCount} active</span>
                            )}
                            {issueCount > 0 && (
                              <span className={cn(statusTone.error.accent, "font-medium")}>{issueCount} issues</span>
                            )}
                          </div>
                        )}
                      </div>

                      <div className="flex items-center min-w-0">
                        <WorkflowHoverCard workflow={workflow}>
                          <div className="flex flex-col min-w-0">
                            <span className="font-semibold truncate text-sm">
                              {workflow.display_name || "Unnamed Workflow"}
                            </span>
                            <span className="text-body-small text-muted-foreground truncate">
                              {workflow.agent_name ?? workflow.agent_id ?? '—'}
                            </span>
                          </div>
                        </WorkflowHoverCard>
                      </div>

                      <div className="flex items-center">
                        <span className="truncate text-body-small">
                          {workflow.current_task}
                        </span>
                      </div>

                      <div className="flex items-center justify-end">
                        <span className="font-medium text-sm">
                          {workflow.total_executions}
                        </span>
                      </div>

                      <div className="flex items-center justify-end">
                        <span className="font-medium text-sm">
                          {activeCount}
                        </span>
                      </div>

                      <div className="flex items-center justify-end">
                        <span className="font-mono text-body-small">
                          {workflow.duration_ms
                            ? formatDurationHumanReadable(workflow.duration_ms)
                            : workflow.status === "running" && workflow.started_at
                              ? <LiveElapsedDuration startedAt={workflow.started_at} className="text-blue-400" />
                              : "—"}
                        </span>
                      </div>

                      <div className="flex items-center justify-center">
                        <WorkflowVCStatusCell
                          workflowId={workflow.workflow_id}
                          summary={workflowVCStatuses[workflow.workflow_id]}
                          loading={workflowVCStatusesLoading}
                        />
                      </div>

                      <div className="flex items-center justify-end">
                        <RelativeTime timestamp={workflow.latest_activity} />
                      </div>

                      <div className="flex items-center justify-end gap-2">
                        <div onClick={(event) => event.stopPropagation()}>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-red-600 hover:text-red-700 hover:bg-red-50"
                            onClick={() => {
                              setSelectedWorkflows(
                                new Set([workflow.run_id])
                              );
                              setShowDeleteDialog(true);
                            }}
                            title="Delete workflow"
                          >
                            <Trash className="w-4 h-4" />
                          </Button>
                        </div>
                        <CaretRight
                          className={cn(
                            "block h-4 w-4 text-muted-foreground",
                            hoveredRow === workflow.run_id
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
              Loading more workflows…
            </div>
          )}
        </CardContent>
      </Card>

      <WorkflowDeleteDialog
        isOpen={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        workflows={selectedWorkflowObjects}
        onConfirm={handleDeleteSelected}
      />
    </div>
  );
}
