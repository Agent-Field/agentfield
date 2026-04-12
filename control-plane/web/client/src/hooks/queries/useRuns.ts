import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { getWorkflowsSummary } from "../../services/workflowsApi";
import type { WorkflowsResponse } from "../../types/workflows";
import { useSSESync } from "../useSSEQuerySync";

export interface RunsFilters {
  timeRange?: string;
  status?: string;
  page?: number;
  pageSize?: number;
  search?: string;
  session?: string;
  actor?: string;
  workflow?: string;
  sortBy?: string;
  sortOrder?: "asc" | "desc";
  /** When set, polls the runs list (e.g. dashboard while workflows are active). */
  refetchInterval?: number | false;
}

export function useRuns(filters: RunsFilters = {}) {
  const { execConnected } = useSSESync();
  const {
    timeRange,
    status,
    page = 1,
    pageSize = 50,
    search,
    session,
    actor,
    workflow,
    sortBy = "latest_activity",
    sortOrder = "desc",
    refetchInterval: explicitRefetch,
  } = filters;

  const refetchInterval = useMemo(() => {
    // Opt-in polling: callers must explicitly pass refetchInterval to auto-refresh.
    // The dashboard passes an interval; list pages (/runs, /agents) get an explicit
    // "Refresh" button instead. This preserves the live-data motion budget contract.
    if (explicitRefetch === undefined) return false;
    if (typeof explicitRefetch === "number") {
      return execConnected ? explicitRefetch : Math.min(explicitRefetch, 5_000);
    }
    return explicitRefetch;
  }, [explicitRefetch, execConnected]);

  return useQuery<WorkflowsResponse>({
    queryKey: ["runs", filters],
    placeholderData: keepPreviousData,
    refetchInterval,
    queryFn: () =>
      getWorkflowsSummary(
        {
          ...(status && status !== "all" ? { status } : {}),
          ...(timeRange && timeRange !== "all" ? { timeRange } : {}),
          ...(search ? { search } : {}),
          ...(session ? { session } : {}),
          ...(actor ? { actor_id: actor } : {}),
          ...(workflow ? { workflow } : {}),
        },
        page,
        pageSize,
        sortBy,
        sortOrder,
      ),
  });
}
