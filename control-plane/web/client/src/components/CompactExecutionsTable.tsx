"use client";


import { useMemo, useState } from "react";
import { Renew, Security, Terminal } from "@/components/ui/icon-bridge";
import { useExecutionVCStatus } from "../hooks/useVCVerification";
import type { EnhancedExecution } from "../types/workflows";
import StatusIndicator from "./ui/status-indicator";
import { VerifiableCredentialBadge } from "./vc/VerifiableCredentialBadge";
import { CompactTable } from "./ui/CompactTable";
import { FastTableSearch, createSearchMatcher } from "./ui/FastTableSearch";
import { useIsMobile } from "../hooks/use-mobile";

// Compact grid layout - reduced from 7 columns to 6 for better spacing
const GRID_TEMPLATE_DESKTOP = "72px minmax(200px,1fr) 120px 80px 64px 120px";
const GRID_TEMPLATE_MOBILE = "40px minmax(140px,1fr) 80px";

interface CompactExecutionsTableProps {
  executions: EnhancedExecution[];
  loading: boolean;
  hasMore: boolean;
  isFetchingMore: boolean;
  sortBy: string;
  sortOrder: "asc" | "desc";
  onSortChange: (field: string, order?: "asc" | "desc") => void;
  onLoadMore?: () => void;
  onExecutionClick?: (execution: EnhancedExecution) => void;
  onRefresh?: () => void;
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

export function CompactExecutionsTable({
  executions,
  loading,
  hasMore,
  isFetchingMore,
  sortBy,
  sortOrder,
  onSortChange,
  onLoadMore,
  onExecutionClick,
  onRefresh,
}: CompactExecutionsTableProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const isMobile = useIsMobile();

  // Define search fields for executions
  const searchFields = [
    "task_name", // Reasoner name
    "agent_name", // Agent name
    "execution_id", // Execution ID
    "status", // Status
    "duration_display", // Duration
  ];

  // Filter executions based on search query
  const filteredExecutions = useMemo(() => {
    if (!searchQuery.trim()) return executions;

    const matcher = createSearchMatcher(searchFields);
    return executions.filter((execution) => matcher(execution, searchQuery));
  }, [executions, searchQuery, searchFields]);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
  };

  const allColumns = [
    {
      key: "status",
      header: isMobile ? "" : "Status",
      sortable: true,
      align: "left" as const,
      render: (execution: EnhancedExecution) => (
        <div className="flex items-center justify-start">
          <StatusIndicator
            status={execution.status as any}
            showLabel={false}
            animated={execution.status === "running"}
          />
        </div>
      ),
    },
    {
      key: "task_name",
      header: "Reasoner",
      sortable: true,
      align: "left" as const,
      render: (execution: EnhancedExecution) => (
        <div className="min-w-0 flex items-center gap-2">
          <span className="truncate text-sm font-medium text-primary-foundation">
            {execution.task_name || "Unknown Reasoner"}
          </span>
          <span className="text-tertiary-foundation text-xs">•</span>
          <div className="flex items-center gap-1.5 text-tertiary-foundation min-w-0">
            <span className="inline-flex h-3 w-3 items-center justify-center rounded-full bg-muted text-[9px] font-medium flex-shrink-0">
              {execution.agent_name
                ? execution.agent_name.charAt(0).toUpperCase()
                : "?"}
            </span>
            <span className="truncate text-xs">
              {execution.agent_name || "Unknown"}
            </span>
          </div>
        </div>
      ),
    },
    {
      key: "when",
      header: "Started",
      sortable: true,
      align: (isMobile ? "right" : "left") as "left" | "right",
      render: (execution: EnhancedExecution) => (
        <div className="text-secondary-foundation timestamp-foundation">
          {execution.relative_time}
        </div>
      ),
    },
    {
      key: "duration",
      header: "Duration",
      sortable: true,
      align: "right" as const,
      render: (execution: EnhancedExecution) => (
        <div className="text-tertiary-foundation execution-id-foundation">
          {execution.duration_display}
        </div>
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
      render: (execution: EnhancedExecution) => (
        <div className="flex items-center justify-center">
          <ExecutionVCStatusCell executionId={execution.execution_id} />
        </div>
      ),
    },
    {
      key: "execution_id",
      header: "Execution ID",
      sortable: true,
      align: "left" as const,
      render: (execution: EnhancedExecution) => (
        <div className="execution-id-foundation">
          …{execution.execution_id.slice(-8)}
        </div>
      ),
    },
  ];

  const columns = useMemo(() => {
    if (isMobile) {
      return allColumns.filter((col) =>
        ["status", "task_name", "when"].includes(col.key)
      );
    }
    return allColumns;
  }, [isMobile]);

  return (
    <div className="space-y-4">
      {/* Search */}
      <FastTableSearch
        onSearch={handleSearch}
        placeholder={isMobile ? "Search executions..." : "Search executions by reasoner, agent, ID, status..."}
        resultCount={filteredExecutions.length}
        totalCount={executions.length}
        disabled={loading}
      />

      {/* Table */}
      <CompactTable
        data={filteredExecutions}
        loading={loading}
        hasMore={hasMore}
        isFetchingMore={isFetchingMore}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onSortChange={onSortChange}
        onLoadMore={onLoadMore}
        onRowClick={onExecutionClick}
        columns={columns}
        gridTemplate={isMobile ? GRID_TEMPLATE_MOBILE : GRID_TEMPLATE_DESKTOP}
        emptyState={{
          title: searchQuery ? "No matching executions" : "No executions yet",
          description: searchQuery
            ? "Try adjusting your search terms or clear the search to see all executions."
            : "Executions will appear here as they run.",
          icon: <Terminal className="h-6 w-6 text-muted-foreground" />,
          action: searchQuery
            ? {
                label: "Clear search",
                onClick: () => setSearchQuery(""),
                icon: <Terminal className="h-3.5 w-3.5" />,
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
        getRowKey={(execution) => execution.execution_id}
      />
    </div>
  );
}

