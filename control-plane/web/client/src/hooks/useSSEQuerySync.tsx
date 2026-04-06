/* eslint-disable react-refresh/only-export-components --
 * Co-located provider + hooks; the provider must wrap the app shell.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  type ReactNode,
} from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useSSE } from "./useSSE";

export type SSESyncContextValue = {
  execConnected: boolean;
  nodeConnected: boolean;
  reasonerConnected: boolean;
  /** True when at least one control-plane SSE channel is connected */
  anyConnected: boolean;
  reconnecting: boolean;
  /** Manual refresh when SSE is down or user wants a full resync */
  refreshAllLiveQueries: () => void;
};

const defaultSSESync: SSESyncContextValue = {
  execConnected: false,
  nodeConnected: false,
  reasonerConnected: false,
  anyConnected: false,
  reconnecting: false,
  refreshAllLiveQueries: () => {},
};

const SSESyncContext = createContext<SSESyncContextValue>(defaultSSESync);

/**
 * Provides execution, node, and reasoner SSE streams plus shared connection state.
 * Mount once at the app shell (e.g. AppLayout).
 */
export function SSESyncProvider({ children }: { children: ReactNode }) {
  const value = useSSEQuerySyncCore();
  return (
    <SSESyncContext.Provider value={value}>{children}</SSESyncContext.Provider>
  );
}

export function useSSESync(): SSESyncContextValue {
  return useContext(SSESyncContext);
}

function shouldInvalidateForEvent(data: unknown): boolean {
  if (!data || typeof data !== "object") return true;

  const type = (data as Record<string, unknown>).type;
  return type !== "connected" && type !== "heartbeat";
}

function useSSEQuerySyncCore(): SSESyncContextValue {
  const queryClient = useQueryClient();

  const {
    latestEvent: execEvent,
    connected: execConnected,
    reconnecting: execReconnecting,
  } = useSSE("/api/ui/v1/executions/events", {
    eventTypes: [
      "execution_completed",
      "execution_failed",
      "execution_started",
      "execution_updated",
      "execution_cancelled",
    ],
    autoReconnect: true,
    maxReconnectAttempts: 3,
    reconnectDelayMs: 3000,
    exponentialBackoff: true,
  });

  useEffect(() => {
    if (!execEvent) return;

    const data =
      execEvent.data && typeof execEvent.data === "object"
        ? (execEvent.data as Record<string, unknown>)
        : {};
    if (!shouldInvalidateForEvent(data)) return;

    void queryClient.invalidateQueries({ queryKey: ["runs"] });
    void queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });

    const runId = (data.run_id ?? data.workflow_id) as string | undefined;
    const execId = data.execution_id as string | undefined;

    if (runId) {
      void queryClient.invalidateQueries({ queryKey: ["run-dag", runId] });
    }
    if (execId) {
      void queryClient.invalidateQueries({ queryKey: ["step-detail", execId] });
    }
  }, [execEvent, queryClient]);

  const {
    latestEvent: nodeEvent,
    connected: nodeConnected,
    reconnecting: nodeReconnecting,
  // Node and reasoner SSE streams are disabled to stay within the browser's
  // HTTP/1.1 per-origin connection limit (6).  Three always-on EventSource
  // connections leave too few slots for API data fetches, causing the UI to
  // hang.  The queries already include adaptive polling fallback when SSE is
  // unavailable, so agent/reasoner updates arrive within a few seconds.
  } = useSSE(null, {
    eventTypes: [
      "node_online",
      "node_offline",
      "node_health_changed",
      "node_unified_status_changed",
      "node_registered",
      "node_removed",
    ],
    autoReconnect: true,
    maxReconnectAttempts: 3,
    reconnectDelayMs: 3000,
    exponentialBackoff: true,
  });

  useEffect(() => {
    if (!nodeEvent) return;

    if (!shouldInvalidateForEvent(nodeEvent.data)) return;

    void queryClient.invalidateQueries({ queryKey: ["agents"] });
    void queryClient.invalidateQueries({ queryKey: ["reasoners"] });
  }, [nodeEvent, queryClient]);

  const {
    latestEvent: reasonerEvent,
    connected: reasonerConnected,
    reconnecting: reasonerReconnecting,
  } = useSSE(null, {
    autoReconnect: true,
    maxReconnectAttempts: 3,
    reconnectDelayMs: 3000,
    exponentialBackoff: true,
  });

  useEffect(() => {
    if (!reasonerEvent) return;
    const data =
      reasonerEvent.data && typeof reasonerEvent.data === "object"
        ? (reasonerEvent.data as Record<string, unknown>)
        : {};
    if (!shouldInvalidateForEvent(data)) return;
    void queryClient.invalidateQueries({ queryKey: ["reasoners"] });
  }, [reasonerEvent, queryClient]);

  const refreshAllLiveQueries = useCallback(() => {
    void queryClient.invalidateQueries({ queryKey: ["runs"] });
    void queryClient.invalidateQueries({ queryKey: ["agents"] });
    void queryClient.invalidateQueries({ queryKey: ["reasoners"] });
    void queryClient.invalidateQueries({ queryKey: ["dashboard-summary"] });
    void queryClient.invalidateQueries({ queryKey: ["run-dag"] });
    void queryClient.invalidateQueries({ queryKey: ["step-detail"] });
    void queryClient.invalidateQueries({ queryKey: ["llm-health"] });
    void queryClient.invalidateQueries({ queryKey: ["queue-status"] });
  }, [queryClient]);

  return useMemo(
    () => ({
      execConnected,
      nodeConnected,
      reasonerConnected,
      anyConnected: execConnected || nodeConnected || reasonerConnected,
      reconnecting:
        execReconnecting || nodeReconnecting || reasonerReconnecting,
      refreshAllLiveQueries,
    }),
    [
      execConnected,
      nodeConnected,
      reasonerConnected,
      execReconnecting,
      nodeReconnecting,
      reasonerReconnecting,
      refreshAllLiveQueries,
    ],
  );
}
