"use client";

import { useMemo, useState, useEffect } from "react";
import type { ReactNode } from "react";
import { Renew, Security, TrashCan } from "@/components/ui/icon-bridge";
import { useWorkflowVCStatuses } from "../hooks/useVCVerification";
import type { WorkflowSummary } from "../types/workflows";
import type { VCStatusSummary } from "../types/did";
import { Button } from "./ui/button";
import { Card, CardContent } from "./ui/card";
import { Checkbox } from "./ui/checkbox";
import { Skeleton } from "./ui/skeleton";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "./ui/hover-card";
import StatusIndicator from "./ui/status-indicator";
import { VerifiableCredentialBadge } from "./vc/VerifiableCredentialBadge";
import { WorkflowDeleteDialog } from "./workflows/WorkflowDeleteDialog";
import {
  deleteWorkflows,
  type WorkflowCleanupResult,
} from "../services/workflowsApi";
import { CompactTable } from "./ui/CompactTable";
import { FastTableSearch, createSearchMatcher } from "./ui/FastTableSearch";
import { normalizeExecutionStatus } from "../utils/status";
import { formatNumber } from "../utils/numberFormat";
import { formatDurationHumanReadable, LiveElapsedDuration } from "@/components/ui/data-formatters";

// Compact grid layout for workflows including selection checkbox
const GRID_TEMPLATE = "40px 140px minmax(200px,1fr) 160px 70px 80px 64px 110px 60px";

interface CompactWorkflowsTableProps {
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
  onRefresh?: () => void;
}

function RelativeTime({ timestamp }: { timestamp: string }) {
  const getRelativeTime = (value: string) => {
    const now = new Date();
    const time = new Date(value);

    // Validate the parsed time
    if (isNaN(time.getTime()) || time.getFullYear() < 1970) {
      return "invalid date";
    }

    const diffMs = now.getTime() - time.getTime();

    // Handle future dates or negative differences
    if (diffMs < 0) {
      return "just now";
    }

    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffMins < 1) return "now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays > 365) return `${Math.floor(diffDays / 365)}y ago`;
    return `${diffDays}d ago`;
  };

  return <span className="timestamp-foundation">{getRelativeTime(timestamp)}</span>;
}

function WorkflowHoverCard({
  workflow,
  children,
}: {
  workflow: WorkflowSummary;
  children: ReactNode;
}) {
  const statusCounts = workflow.status_counts ?? {};
  const activeExecutions = workflow.active_executions ?? 0;
  const failedExecutions = (statusCounts.failed ?? 0) + (statusCounts.timeout ?? 0);
  const agentLabel = workflow.agent_name ?? workflow.agent_id ?? '—';

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
              <div className="font-medium" title={workflow.total_executions.toLocaleString()}>
                {formatNumber(workflow.total_executions)}
              </div>
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
              <div className="font-medium" title={activeExecutions.toLocaleString()}>
                {formatNumber(activeExecutions)}
              </div>
            </div>
            <div>
              <span className="text-muted-foreground">Succeeded:</span>
              <div className="font-medium" title={(statusCounts.succeeded ?? 0).toLocaleString()}>
                {formatNumber(statusCounts.succeeded ?? 0)}
              </div>
            </div>
            <div>
              <span className="text-muted-foreground">Failed:</span>
              <div className="font-medium text-destructive" title={failedExecutions.toLocaleString()}>
                {formatNumber(failedExecutions)}
              </div>
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
    return <Skeleton className="h-3 w-12 rounded-full bg-muted/20" />;
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

export function CompactWorkflowsTable({
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
  onRefresh,
}: CompactWorkflowsTableProps) {
  const [selectedWorkflows, setSelectedWorkflows] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const workflowIds = useMemo(
    () => workflows.map((workflow) => workflow.workflow_id).filter(Boolean),
    [workflows]
  );
  const { statuses: workflowVCStatuses, loading: workflowVCStatusesLoading } =
    useWorkflowVCStatuses(workflowIds);

  // Define search fields for workflows
  const searchFields = [
    "display_name",    // Workflow name
    "agent_name",      // Agent name
    "current_task",    // Last reasoner
    "workflow_id",     // Workflow ID
    "run_id",          // Run ID
    "status",          // Status
    "root_reasoner",   // Root task
  ];

  // Filter workflows based on search query
  const filteredWorkflows = useMemo(() => {
    if (!searchQuery.trim()) return workflows;

    const matcher = createSearchMatcher(searchFields);
    return workflows.filter(workflow => matcher(workflow, searchQuery));
  }, [workflows, searchQuery, searchFields]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

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

  const getStatusCounts = (workflow: WorkflowSummary) => {
    if (workflow.status_counts) {
      return workflow.status_counts;
    }
    return {};
  };

  const getActiveExecutions = (workflow: WorkflowSummary) => {
    if (typeof workflow.active_executions === "number") {
      return workflow.active_executions;
    }
    const counts = getStatusCounts(workflow);
    return (
      (counts.running ?? 0) +
      (counts.queued ?? 0) +
      (counts.pending ?? 0)
    );
  };

  const columns = [
    {
      key: "select",
      header: (
        <div className="flex items-center justify-center">
          <Checkbox
            checked={isAllSelected}
            onCheckedChange={(value) => handleSelectAll(Boolean(value))}
            aria-label="Select all workflows"
            className={isPartiallySelected ? "data-[state=checked]:bg-primary/50" : ""}
          />
        </div>
      ),
      sortable: false,
      align: "center" as const,
      render: (workflow: WorkflowSummary) => {
        const isSelected = selectedWorkflows.has(workflow.run_id);
        return (
          <div
            className="flex items-center justify-center"
            onClick={(event) => event.stopPropagation()}
          >
            <Checkbox
              checked={isSelected}
              onCheckedChange={(value) =>
                handleSelectWorkflow(workflow.run_id, Boolean(value))
              }
              aria-label={`Select workflow ${workflow.display_name}`}
            />
          </div>
        );
      },
    },
    {
      key: "status",
      header: "Status",
      sortable: true,
      align: "left" as const,
      render: (workflow: WorkflowSummary) => {
        const normalized = normalizeExecutionStatus(workflow.status);
        const active = getActiveExecutions(workflow);
        const counts = getStatusCounts(workflow);
        const failed = (counts.failed ?? 0) + (counts.timeout ?? 0);
        const succeeded = counts.succeeded ?? 0;

        // Smart status display - show most relevant information
        const getStatusDisplay = () => {
          // For active workflows, prioritize showing active count
          if (active > 0) {
            return {
              primary: `${formatNumber(active)} active`,
              secondary: failed > 0 ? `${formatNumber(failed)} issue${failed === 1 ? '' : 's'}` : null,
              primaryClass: "status-active",
              secondaryClass: "status-issues"
            };
          }

          // For failed workflows, show issues
          if (failed > 0) {
            return {
              primary: `${formatNumber(failed)} issue${failed === 1 ? '' : 's'}`,
              secondary: null,
              primaryClass: "status-issues",
              secondaryClass: ""
            };
          }

          // For completed workflows, show completion count
          if (normalized === "succeeded" && succeeded > 0) {
            return {
              primary: `${formatNumber(succeeded)} completed`,
              secondary: null,
              primaryClass: "status-completed",
              secondaryClass: ""
            };
          }

          // For other states, show status name
          const statusLabels: Record<string, string> = {
            "running": "Running",
            "queued": "Queued",
            "pending": "Pending",
            "cancelled": "Cancelled",
            "timeout": "Timeout",
            "succeeded": "Completed",
            "failed": "Failed"
          };

          return {
            primary: statusLabels[normalized] || "Unknown",
            secondary: null,
            primaryClass: "status-neutral",
            secondaryClass: ""
          };
        };

        const statusDisplay = getStatusDisplay();

        return (
          <div className="flex items-center gap-2 py-1">
            <StatusIndicator
              status={normalized}
              showLabel={false}
              animated={normalized === "running" || normalized === "queued"}
            />
            <div className="flex items-center gap-2 min-w-0">
              <span className={`text-sm font-medium ${statusDisplay.primaryClass}`}>
                {statusDisplay.primary}
              </span>
              {statusDisplay.secondary && (
                <>
                  <span className="text-body-small">•</span>
                  <span className={`text-xs font-medium ${statusDisplay.secondaryClass}`}>
                    {statusDisplay.secondary}
                  </span>
                </>
              )}
            </div>
          </div>
        );
      },
    },
    {
      key: "workflow_name",
      header: "Workflow",
      sortable: false,
      align: "left" as const,
      render: (workflow: WorkflowSummary) => (
        <div className="min-w-0">
          <WorkflowHoverCard workflow={workflow}>
            <div className="flex items-center gap-2 min-w-0">
              <span className="font-medium truncate text-primary-foundation text-sm">
                {workflow.display_name || "Unnamed Workflow"}
              </span>
              <span className="text-tertiary-foundation text-xs">•</span>
              <span className="text-tertiary-foundation truncate text-xs">
                {workflow.agent_name ?? workflow.agent_id ?? '—'}
              </span>
            </div>
          </WorkflowHoverCard>
        </div>
      ),
    },
    {
      key: "current_task",
      header: "Last Reasoner",
      sortable: false,
      align: "left" as const,
      render: (workflow: WorkflowSummary) => (
        <span className="truncate text-secondary-foundation">
          {workflow.current_task}
        </span>
      ),
    },
    {
      key: "total_executions",
      header: "Nodes",
      sortable: true,
      align: "right" as const,
      render: (workflow: WorkflowSummary) => (
        <span className="text-primary-foundation font-medium" title={workflow.total_executions.toLocaleString()}>
          {formatNumber(workflow.total_executions)}
        </span>
      ),
    },
    {
      key: "duration",
      header: "Duration",
      sortable: false,
      align: "right" as const,
      render: (workflow: WorkflowSummary) => (
        <span className="text-tertiary-foundation execution-id-foundation">
          {workflow.duration_ms
            ? formatDurationHumanReadable(workflow.duration_ms)
            : workflow.status === "running" && workflow.started_at
              ? <LiveElapsedDuration startedAt={workflow.started_at} className="text-blue-400" />
              : "—"}
        </span>
      ),
    },
    {
      key: "vc",
      header: (
        <div className="flex items-center justify-center gap-1">
          <Security className="h-3 w-3" />
          <span>VC</span>
        </div>
      ),
      sortable: false,
      align: "center" as const,
      render: (workflow: WorkflowSummary) => (
        <WorkflowVCStatusCell
          workflowId={workflow.workflow_id}
          summary={workflowVCStatuses[workflow.workflow_id]}
          loading={workflowVCStatusesLoading}
        />
      ),
    },
    {
      key: "latest_activity",
      header: "Updated",
      sortable: true,
      align: "right" as const,
      render: (workflow: WorkflowSummary) => (
        <RelativeTime timestamp={workflow.latest_activity} />
      ),
    },
    {
      key: "actions",
      header: "",
      sortable: false,
      align: "center" as const,
      render: (workflow: WorkflowSummary) => (
        <div onClick={(event) => event.stopPropagation()}>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-red-600 hover:text-red-700 hover:bg-red-50"
            onClick={() => {
              setSelectedWorkflows(new Set([workflow.run_id]));
              setShowDeleteDialog(true);
            }}
            title="Delete workflow"
          >
            <TrashCan className="w-3 h-3" />
          </Button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-4">
      {/* Search */}
      <FastTableSearch
        onSearch={handleSearch}
        placeholder="Search workflows by name, agent, reasoner, status..."
        resultCount={filteredWorkflows.length}
        totalCount={workflows.length}
        disabled={loading}
      />

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
                <TrashCan className="w-4 h-4" />
                Delete Selected
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <CompactTable
        data={filteredWorkflows}
        loading={loading}
        hasMore={hasMore}
        isFetchingMore={isFetchingMore}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onSortChange={onSortChange}
        onLoadMore={onLoadMore}
        onRowClick={onWorkflowClick}
        columns={columns}
        gridTemplate={GRID_TEMPLATE}
        emptyState={{
          title: searchQuery ? "No matching workflows" : "No workflows yet",
          description: searchQuery
            ? "Try adjusting your search terms or clear the search to see all workflows."
            : "Workflows will appear here as they execute.",
          icon: <Security className="h-6 w-6 text-muted-foreground" />,
          action: searchQuery
            ? {
                label: "Clear search",
                onClick: () => setSearchQuery(""),
                icon: <Security className="h-3.5 w-3.5" />,
              }
            : onRefresh
              ? {
                label: loading ? "Fetching…" : "Refresh data",
                onClick: onRefresh,
                icon: <Renew className="h-3.5 w-3.5" />,
              }
              : undefined,
          secondaryAction:
            searchQuery && onRefresh
              ? {
                label: loading ? "Fetching…" : "Refresh data",
                onClick: onRefresh,
                icon: <Renew className="h-3.5 w-3.5" />,
              }
              : undefined,
        }}
        getRowKey={(workflow) => workflow.run_id}
      />

      <WorkflowDeleteDialog
        isOpen={showDeleteDialog}
        onClose={() => setShowDeleteDialog(false)}
        workflows={selectedWorkflowObjects}
        onConfirm={handleDeleteSelected}
      />
    </div>
  );
}
