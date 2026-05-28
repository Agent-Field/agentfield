import { useQuery } from "@tanstack/react-query";
import { useMemo } from "react";
import { getNodesSummary } from "../../services/api";
import type { AgentNodeSummary } from "../../types/agentfield";
import { useSSESync } from "../useSSEQuerySync";

interface NodesSummaryResponse {
  nodes: AgentNodeSummary[];
  count: number;
}

interface UseAgentsOptions {
  /** Opt-in polling interval. When omitted, the query does not auto-refresh. */
  refetchInterval?: number | false;
}

export function useAgents(options: UseAgentsOptions = {}) {
  const { nodeConnected } = useSSESync();
  const { refetchInterval: explicitRefetch } = options;

  const refetchInterval = useMemo(() => {
    if (explicitRefetch === undefined) return false;
    if (typeof explicitRefetch === "number") {
      return nodeConnected ? explicitRefetch : Math.min(explicitRefetch, 5_000);
    }
    return explicitRefetch;
  }, [explicitRefetch, nodeConnected]);

  return useQuery<NodesSummaryResponse>({
    queryKey: ["agents"],
    queryFn: () => getNodesSummary(),
    refetchInterval,
  });
}
